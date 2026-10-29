// The caller's max_tokens, temperature and top_p reach Kiro. They were
// dropped: the request kept the conversation, the agent mode and the profile
// and nothing else, though the models declare temperature. Kiro's
// generateAssistantResponse takes them beside conversationState as
// inferenceConfig ({maxTokens, temperature, topP}, Bedrock Converse's
// InferenceConfiguration), as other Kiro clients send them; kiro-cli 2.35
// sends none. Only what the caller set goes, within Anthropic's bounds. Claude
// Sonnet 4.5 and Haiku 4.5 turn away temperature and top_p together
// (Bedrock's Claude request docs), so a Claude model, or Auto, is sent the
// temperature alone when both are set.
import { expect, test } from "bun:test"
import { _internal } from "./index.mjs"

const { buildKiro, kiroBody, generate } = _internal

const ask = (extra) => ({ model: "claude-sonnet-4.5", messages: [{ role: "user", content: "hi" }], ...extra })
const built = (extra, model = "claude-sonnet-4.5") => JSON.parse(buildKiro(ask(extra), model, "", 0))

test("max_tokens, temperature and top_p go as inferenceConfig", async () => {
  const set = { max_tokens: 1024, temperature: 0.2, top_p: 0.9 }
  const want = { maxTokens: 1024, temperature: 0.2, topP: 0.9 }
  expect(built(set, "deepseek-3.2").inferenceConfig).toEqual(want)
  expect(JSON.parse(await kiroBody(ask(set), "deepseek-3.2", "", 0)).inferenceConfig).toEqual(want)
  expect(JSON.parse(await kiroBody(ask(set), "claude-sonnet-4.5", "", 0)).inferenceConfig).toEqual({ maxTokens: 1024, temperature: 0.2 })
})

test("a Claude model, or Auto, is sent the temperature alone when both are set", () => {
  for (const model of ["claude-sonnet-4.5", "claude-haiku-4.5", "CLAUDE-OPUS-4.5", "auto"]) {
    expect(built({ temperature: 0.7, top_p: 0.9 }, model).inferenceConfig).toEqual({ temperature: 0.7 })
    expect(built({ temperature: 0, top_p: 0 }, model).inferenceConfig).toEqual({ temperature: 0 })
    expect(built({ top_p: 0.9 }, model).inferenceConfig).toEqual({ topP: 0.9 })
    // a temperature out of bounds isn't sent, so top_p is
    expect(built({ temperature: 2, top_p: 0.9 }, model).inferenceConfig).toEqual({ topP: 0.9 })
    expect(built({ temperature: "0.7", top_p: 0 }, model).inferenceConfig).toEqual({ topP: 0 })
  }
  expect(built({ temperature: 0.7, top_p: 0.9 }, "deepseek-3.2").inferenceConfig).toEqual({ temperature: 0.7, topP: 0.9 })
})

test("a request that sets none sends no inferenceConfig", () => {
  expect(built({})).not.toHaveProperty("inferenceConfig")
  expect(built({ max_tokens: undefined, temperature: null })).not.toHaveProperty("inferenceConfig")
})

test("only the fields set are sent, zero included", () => {
  expect(built({ max_tokens: 1 }).inferenceConfig).toEqual({ maxTokens: 1 })
  expect(built({ temperature: 0 }).inferenceConfig).toEqual({ temperature: 0 })
  expect(built({ top_p: 0 }).inferenceConfig).toEqual({ topP: 0 })
  expect(built({ temperature: 1, top_p: 1 }, "deepseek-3.2").inferenceConfig).toEqual({ temperature: 1, topP: 1 })
})

test("a value out of bounds or not a number is left out", () => {
  for (const max_tokens of [0, -1, 1.5, NaN, Infinity, "1024", 2 ** 53])
    expect(built({ max_tokens, temperature: 0.5 }).inferenceConfig).toEqual({ temperature: 0.5 })
  for (const v of [-0.1, 1.1, NaN, Infinity, -Infinity, "0.5", true])
    for (const model of ["claude-sonnet-4.5", "deepseek-3.2"])
      expect(built({ max_tokens: 64, temperature: v, top_p: v }, model).inferenceConfig).toEqual({ maxTokens: 64 })
  expect(built({ max_tokens: 0, temperature: 2, top_p: -1 })).not.toHaveProperty("inferenceConfig")
})

test("thinking keeps the caller's settings and its own budget", () => {
  const body = JSON.parse(buildKiro(ask({ max_tokens: 64000, temperature: 1, thinking: { type: "enabled", budget_tokens: 20000 } }), "claude-sonnet-4.5", "", 20000))
  expect(body.inferenceConfig).toEqual({ maxTokens: 64000, temperature: 1 })
  expect(body.conversationState.currentMessage.userInputMessage.content).toStartWith("<thinking_mode>enabled</thinking_mode><max_thinking_length>20000</max_thinking_length>")
})

test("Kiro is sent them, beside the conversation", async () => {
  const real = globalThis.fetch
  const sent = []
  globalThis.fetch = async (url, init) => {
    sent.push({ url: String(url), body: JSON.parse(init.body) })
    return new Response(JSON.stringify({ message: "stop here" }), { status: 500 })
  }
  try {
    const creds = async () => ({ region: "us-east-1", token: "t", profile: "arn:aws:codewhisperer:us-east-1:1:profile/P" })
    await (await generate(creds, {}, ask({ max_tokens: 2048, temperature: 0, top_p: 0.5, stream: true }))).text()
    await (await generate(creds, {}, ask({}))).text()
    await (await generate(creds, {}, { ...ask({ max_tokens: 2048, temperature: 0, top_p: 0.5 }), model: "deepseek-3.2" })).text()
  } finally {
    globalThis.fetch = real
  }
  expect(sent).toHaveLength(3)
  expect(sent[0].url).toBe("https://runtime.us-east-1.kiro.dev/generateAssistantResponse")
  expect(Object.keys(sent[0].body).sort()).toEqual(["agentMode", "conversationState", "inferenceConfig", "profileArn"])
  expect(sent[0].body.inferenceConfig).toEqual({ maxTokens: 2048, temperature: 0 })
  expect(sent[0].body.conversationState.currentMessage.userInputMessage.modelId).toBe("claude-sonnet-4.5")
  expect(sent[1].body).not.toHaveProperty("inferenceConfig")
  expect(sent[2].body.inferenceConfig).toEqual({ maxTokens: 2048, temperature: 0, topP: 0.5 })
  expect(sent[2].body.conversationState.currentMessage.userInputMessage.modelId).toBe("deepseek-3.2")
})
