import { useMemo, useRef, useState } from "react";
import {
  Check,
  Copy,
  Cpu,
  Loader2,
  MoonStar,
  RefreshCw,
  Send,
  Share2,
  Sparkles,
  Wand2,
} from "lucide-react";
import {
  CreateWebWorkerMLCEngine,
  type MLCEngineInterface,
} from "@mlc-ai/web-llm";

const MODEL_OPTIONS = [
  {
    id: "gemini-nano",
    name: "Gemini Nano",
    badge: "Chrome",
    note: "Chrome組み込みAI。対応ChromeならモデルDL後にブラウザ内で動作",
    vram: "内蔵",
    backend: "gemini-nano",
  },
  {
    id: "Qwen3-1.7B-q4f16_1-MLC",
    name: "Qwen3 1.7B",
    badge: "推奨",
    note: "動作確認済み。速度と安定性の本命",
    vram: "約2.1GB",
    backend: "web-llm",
  },
  {
    id: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
    name: "Llama 3.2 1B",
    badge: "軽量",
    note: "0.8Bが不安定な時の1.7B未満候補",
    vram: "約0.9GB",
    backend: "web-llm",
  },
  {
    id: "Qwen3.5-2B-q4f16_1-MLC",
    name: "Qwen3.5 2B",
    badge: "品質",
    note: "少し重いが文章品質を上げたい時に",
    vram: "約2.3GB",
    backend: "web-llm",
  },
] as const;

type ModelOption = (typeof MODEL_OPTIONS)[number];
type ModelBackend = ModelOption["backend"];

type BuiltInLanguageModelAvailability = "available" | "downloadable" | "downloading" | "unavailable";

type BuiltInLanguageModelSession = {
  prompt(input: string): Promise<string>;
  destroy?(): void;
};

type BuiltInLanguageModelMonitor = {
  addEventListener(
    type: "downloadprogress",
    listener: (event: { loaded: number }) => void,
  ): void;
};

type BuiltInLanguageModelFactory = {
  availability(options?: unknown): Promise<BuiltInLanguageModelAvailability>;
  create(options?: {
    monitor?: (monitor: BuiltInLanguageModelMonitor) => void;
    expectedInputs?: Array<{ type: "text"; languages: string[] }>;
    expectedOutputs?: Array<{ type: "text"; languages: string[] }>;
  }): Promise<BuiltInLanguageModelSession>;
};

declare global {
  interface Window {
    LanguageModel?: BuiltInLanguageModelFactory;
  }
}

const RESPONSE_COUNT = 1;

const MAJI_RES_SCHEMA = JSON.stringify({
  type: "array",
  minItems: RESPONSE_COUNT,
  maxItems: RESPONSE_COUNT,
  items: {
    type: "string",
    minLength: 1,
    maxLength: 60,
  },
});

const ANALYSIS_SCHEMA = JSON.stringify({
  type: "object",
  additionalProperties: false,
  required: ["category", "target", "reality", "metric", "avoid", "angle"],
  properties: {
    category: { type: "string" },
    target: { type: "string" },
    reality: { type: "string" },
    metric: { type: "string" },
    avoid: { type: "string" },
    angle: { type: "string" },
  },
});

type AppMode = "input" | "loading" | "result" | "error";

type InitProgress = {
  progress?: number;
  text?: string;
};

type EngineSlot = {
  modelId: string;
  worker: Worker;
  promise: Promise<MLCEngineInterface>;
};

type WishAnalysis = {
  category: string;
  target: string;
  reality: string;
  metric: string;
  avoid: string;
  angle: string;
};

const sampleWishes = [
  "今年中に副業で月10万円稼ぎたい",
  "英語を話せるようになりたい",
  "健康的に5kg痩せたい",
];

function buildAnalysisPrompt(wishText: string) {
  return `願い事を分析してください。まだマジレス本文は作らないでください。

# 制約事項
- JSONオブジェクトのみ出力してください。
- category は「健康」「仕事」「学習」「お金」「恋愛」「創作」「生活」「その他」から選んでください。
- target は願いの具体的な目標を短く書いてください。
- reality はその願いで避けられない現実の壁を短く書いてください。
- metric は現実を測る数字・記録・期限を短く書いてください。
- avoid は生成時に避けるべき薄い表現や変な比喩を書いてください。
- angle は最終マジレスで突くべき切り口を短く書いてください。

# 願い事
「${wishText}」`;
}

function buildMajiResPrompt(wishText: string, analysis: WishAnalysis) {
  return `# あなたの役割
あなたは、1年に1度しか願いを聞かない、現実主義で超辛口な「マジレス短冊AI」です。
願い事と分析結果をもとに、笑えるけど痛い【最高のマジレスを1つだけ】返してください。

# 願い事
「${wishText}」

# 分析結果
- category: ${analysis.category}
- target: ${analysis.target}
- reality: ${analysis.reality}
- metric: ${analysis.metric}
- avoid: ${analysis.avoid}
- angle: ${analysis.angle}

# 制約事項
- 回答は必ずJSON配列のみで出力してください。
- 配列の要素数は必ず1つです。
- 配列の中身は日本語の文字列1つだけです。
- 60文字以内にしてください。
- 願い事の言い換えや肯定だけは禁止です。
- analysis.reality か analysis.metric の内容を必ず反映してください。
- 比喩、動物たとえ、外見いじり、不自然な日本語は禁止です。
- 「〜に似ている」「〜ということだ」のような説明調は禁止です。
- 行動ステップ、TODO、ロードマップにしないでください。
- 薄い励ましではなく、短冊に書ける一撃のツッコミにしてください。

# 口調
辛口、現実的、ちょっと笑える、でもどこか愛がある。長く説明しない。

# 出力JSONの構造
- トップレベルは配列
- 配列の要素は1個
- 各要素は文字列
- オブジェクトは禁止`;
}

function parseAnalysis(rawText: string): WishAnalysis {
  const start = rawText.indexOf("{");
  const end = rawText.lastIndexOf("}");

  if (start === -1 || end === -1 || end <= start) {
    throw new Error("分析JSONが見つかりませんでした。");
  }

  const parsed = JSON.parse(rawText.slice(start, end + 1));
  const analysis = {
    category: String(parsed.category ?? "").trim(),
    target: String(parsed.target ?? "").trim(),
    reality: String(parsed.reality ?? "").trim(),
    metric: String(parsed.metric ?? "").trim(),
    avoid: String(parsed.avoid ?? "").trim(),
    angle: String(parsed.angle ?? "").trim(),
  };

  if (
    !analysis.category ||
    !analysis.target ||
    !analysis.reality ||
    !analysis.metric ||
    !analysis.avoid ||
    !analysis.angle
  ) {
    throw new Error("分析JSONの項目が不足しています。");
  }

  return analysis;
}

function createLocalAnalysis(wishText: string): WishAnalysis {
  const normalizedWish = wishText.replace(/\s/g, "");

  if (/痩せ|減量|ダイエット|体重|健康/.test(normalizedWish)) {
    return {
      category: "健康",
      target: "健康的に体重を落とす",
      reality: "食事記録と摂取カロリーから逃げられない",
      metric: "体重、間食、カロリー、睡眠",
      avoid: "体重を捨てる、動物たとえ、外見いじり",
      angle: "願望より毎日の記録を突く",
    };
  }

  if (/大手|企業|就職|転職|会社|仕事|年収|キャリア/.test(normalizedWish)) {
    return {
      category: "仕事",
      target: "仕事やキャリアを良くする",
      reality: "競争相手と実績で比べられる",
      metric: "成果、数字、職務経歴、面接",
      avoid: "夢は大事、頑張れば叶う",
      angle: "願望より面接で語れる数字を突く",
    };
  }

  if (/英語|語学|勉強|資格|試験|受験|合格|話せ/.test(normalizedWish)) {
    return {
      category: "学習",
      target: "勉強や語学を身につける",
      reality: "毎日の練習時間とテスト結果が必要",
      metric: "学習時間、過去問、点数、継続",
      avoid: "いつか話せる、気持ちはある",
      angle: "気持ちより今日の練習時間を突く",
    };
  }

  if (/稼|副業|起業|収入|お金|売上/.test(normalizedWish)) {
    return {
      category: "お金",
      target: "収入や売上を増やす",
      reality: "誰に何円で売るか決める必要がある",
      metric: "売上、単価、顧客数、継続",
      avoid: "楽して稼ぐ、夢がある",
      angle: "願望より売上ゼロの現実を突く",
    };
  }

  return {
    category: "その他",
    target: "願いを現実に近づける",
    reality: "期限と行動記録がないと進まない",
    metric: "期限、行動ログ、継続",
    avoid: "願いはきれい、夢は大事",
    angle: "無期限の願いを突く",
  };
}

function parseMajiRes(rawText: string, wishText: string): string[] {
  const start = rawText.indexOf("[");
  const end = rawText.lastIndexOf("]");

  if (start === -1 || end === -1 || end <= start) {
    throw new Error("JSON配列が見つかりませんでした。");
  }

  const parsed = JSON.parse(rawText.slice(start, end + 1));

  if (!Array.isArray(parsed) || parsed.length !== RESPONSE_COUNT) {
    throw new Error("1つの配列ではありませんでした。");
  }

  const placeholderFragments = [
    "辛口な現実の指摘",
    "次の課題の提示",
    "中だるみへの警告",
    "ターニングポイント",
    "実践的な追い込み",
    "仕上げ・検証",
    "目標達成・継続の喝",
    "具体的な行動",
    "辛口コメント",
    "1つ目",
    "ほんの少しの願い",
    "頑張れば叶う",
    "夢は大事",
    "願いだ",
    "体重を捨てる",
    "痩せは",
    "似ている",
    "ということだ",
    "クマ",
    "動物",
    "足のサイズ",
    "教えてくれ",
    "私の足",
    "その体重は",
    "痩せたなら",
  ];

  const realityFragments = [
    "競争",
    "時間",
    "実績",
    "面接",
    "数字",
    "継続",
    "食事",
    "カロリー",
    "運動",
    "睡眠",
    "記録",
    "体重",
    "間食",
    "歩",
    "練習",
    "テスト",
    "倍率",
    "職務経歴",
    "ポートフォリオ",
    "期限",
    "毎日",
    "現実",
  ];

  const replies = parsed.map((item) => {
    if (typeof item !== "string") {
      throw new Error("文字列配列ではありませんでした。");
    }

    const reply = item.trim();

    if (!reply) {
      throw new Error("空のマジレスが含まれていました。");
    }

    return reply.slice(0, 60);
  });

  const copiedTemplate = replies.some((reply) =>
    placeholderFragments.some((fragment) => reply.includes(fragment)),
  );

  if (copiedTemplate) {
    throw new Error("フォーマット例の丸写しでした。");
  }

  const tooCloseToWish = replies.some((reply) => {
    const normalizedReply = reply.replace(/[！!？?\s。、,.]/g, "");
    const normalizedWish = wishText.replace(/[！!？?\s。、,.]/g, "");

    return normalizedWish.length >= 8 && normalizedReply.includes(normalizedWish);
  });

  const hasReality = replies.some((reply) =>
    realityFragments.some((fragment) => reply.includes(fragment)),
  );

  if (tooCloseToWish || !hasReality) {
    throw new Error("願いの言い換えだけで、現実の壁がありませんでした。");
  }

  return replies;
}

function createFallbackMajiRes(wishText: string, analysis = createLocalAnalysis(wishText)) {
  const normalizedWish = wishText.replace(/\s/g, "");
  const pick = (messages: string[]) => {
    const seed = Array.from(normalizedWish).reduce(
      (total, char) => total + char.charCodeAt(0),
      normalizedWish.length,
    );

    return messages[seed % messages.length];
  };

  if (/痩せ|減量|ダイエット|体重|健康/.test(normalizedWish)) {
    return pick([
      "5kgは願いじゃなくて記録の積み上げ。まず間食とカロリーから逃げるな。",
      "健康的に痩せたいなら、体重より先に食事記録から逃げるな。",
      "運動より先に、昨日の摂取カロリーを言えない時点で現実が勝ってる。",
    ]);
  }

  if (/大手|企業|就職|転職|会社|仕事|年収|キャリア/.test(normalizedWish)) {
    return pick([
      "大手は願望より実績を見る。面接で語れる数字を先に作れ。",
      "入りたい会社名より、職務経歴に書ける成果を先に増やせ。",
      "大手志望は自由だが、競争相手も同じ願いを持ってるぞ。",
    ]);
  }

  if (/英語|語学|勉強|資格|試験|受験|合格|話せ/.test(normalizedWish)) {
    return pick([
      "話せるようになりたいなら、毎日の練習時間を先に固定しろ。",
      "資格は願いで受からない。過去問の点数だけが現実を動かす。",
      "勉強したい気持ちは偉い。で、今日の学習時間は何分だ。",
    ]);
  }

  if (/稼|副業|起業|収入|お金|売上/.test(normalizedWish)) {
    return pick([
      "稼ぎたいなら願うより売れ。数字が出るまでただの趣味だ。",
      "副業の敵は才能じゃない。売上ゼロでも続ける地味さだ。",
      "収入を増やしたいなら、まず誰に何円で売るか決めろ。",
    ]);
  }

  if (/恋|結婚|彼氏|彼女|モテ|出会/.test(normalizedWish)) {
    return pick([
      "出会いが欲しいなら、待機時間より外に出る回数を増やせ。",
      "恋は奇跡待ちじゃない。返信と清潔感と予定調整の現実だ。",
      "理想の相手を語る前に、自分が選ばれる理由を一つ作れ。",
    ]);
  }

  if (/幸せ|楽|自由|人生|変え|成功|夢|叶/.test(normalizedWish)) {
    return pick([
      "人生を変えたいなら、まず今日の時間の使い方を変えろ。",
      "自由が欲しいなら、先に数字と責任から逃げるのをやめろ。",
      "夢はきれいだが、期限がないならただの気分転換だ。",
    ]);
  }

  if (/筋肉|筋トレ|強く|体力|運動/.test(normalizedWish)) {
    return pick([
      "筋肉は願いを聞かない。週何回やったかだけ覚えてる。",
      "強くなりたいなら、まず睡眠と継続をサボる言い訳を捨てろ。",
      "体力は短冊じゃ増えない。今日歩いた歩数が現実だ。",
    ]);
  }

  if (/絵|歌|小説|漫画|作品|上手|クリエイ|作/.test(normalizedWish)) {
    return pick([
      "上手くなりたいなら、才能談義より完成数を増やせ。",
      "作品は構想じゃ評価されない。公開した数だけが現実だ。",
      "創作の敵はセンス不足じゃない。未完成を抱える癖だ。",
    ]);
  }

  return pick([
    `${analysis.target}なら、まず${analysis.metric}から逃げるな。`,
    `${analysis.reality}。短冊より先に今日の証拠を出せ。`,
    `${analysis.angle}。願いより${analysis.metric}を見せろ。`,
  ]);
}

export default function App() {
  const [wishText, setWishText] = useState("");
  const [selectedModelId, setSelectedModelId] = useState<string>("Qwen3-1.7B-q4f16_1-MLC");
  const [mode, setMode] = useState<AppMode>("input");
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState("待機中");
  const [majiResReplies, setMajiResReplies] = useState<string[]>([]);
  const [errorMessage, setErrorMessage] = useState("");
  const [shareStatus, setShareStatus] = useState("");
  const engineSlotRef = useRef<EngineSlot | null>(null);

  const progressPercent = useMemo(() => Math.round(progress * 100), [progress]);
  const selectedModel = MODEL_OPTIONS.find((model) => model.id === selectedModelId) ?? MODEL_OPTIONS[0];

  async function getEngine(modelId: string) {
    if (engineSlotRef.current?.modelId === modelId) {
      return engineSlotRef.current.promise;
    }

    engineSlotRef.current?.worker.terminate();

    const worker = new Worker(new URL("./worker.ts", import.meta.url), {
      type: "module",
    });

    const promise = CreateWebWorkerMLCEngine(worker, modelId, {
      initProgressCallback: (report: InitProgress) => {
        setProgress(Math.max(0, Math.min(1, report.progress ?? 0)));
        setProgressText(report.text ?? "AIを読み込み中");
      },
    });

    engineSlotRef.current = { modelId, worker, promise };
    return promise;
  }

  async function promptGeminiNano(prompt: string) {
    const languageModel = window.LanguageModel;

    if (!languageModel) {
      throw new Error("Gemini Nanoを使うChrome Prompt APIが見つかりません。");
    }

    const languageOptions = {
      expectedInputs: [{ type: "text" as const, languages: ["ja"] }],
      expectedOutputs: [{ type: "text" as const, languages: ["ja"] }],
    };
    const availability = await languageModel.availability(languageOptions);

    if (availability === "unavailable") {
      throw new Error("この環境ではGemini Nanoを利用できません。Chromeの対応版で試してください。");
    }

    const session = await languageModel.create({
      ...languageOptions,
      monitor(monitor) {
        monitor.addEventListener("downloadprogress", (event) => {
          setProgress(Math.max(0, Math.min(1, event.loaded ?? 0)));
          setProgressText("Gemini Nanoをダウンロード中");
        });
      },
    });

    try {
      return await session.prompt(prompt);
    } finally {
      session.destroy?.();
    }
  }

  async function generateWithGeminiNano(cleanWish: string) {
    let analysis = createLocalAnalysis(cleanWish);
    let lastReply = "";

    setProgressText("Gemini Nanoで願い事を分解中");

    try {
      const analysisReply = await promptGeminiNano(
        `${buildAnalysisPrompt(cleanWish)}\n\nJSONオブジェクトのみを出力してください。`,
      );
      analysis = parseAnalysis(analysisReply);
    } catch (error) {
      console.warn("Using local wish analysis after invalid Gemini Nano output:", error);
    }

    setProgressText("Gemini Nanoで辛口の一撃を生成中");

    for (let attempt = 0; attempt < 2; attempt += 1) {
      lastReply = await promptGeminiNano(
        `${buildMajiResPrompt(cleanWish, analysis)}\n\n${
          attempt === 0
            ? "分析結果を使って、最高の辛口マジレスを1つだけ作ってください。"
            : "前回は形式崩れか品質不足でした。新しい1要素の文字列JSON配列だけを出力してください。"
        }`,
      );

      try {
        return parseMajiRes(lastReply, cleanWish);
      } catch {
        continue;
      }
    }

    console.warn("Using fallback maji-res after invalid Gemini Nano output:", lastReply);
    return [createFallbackMajiRes(cleanWish, analysis)];
  }

  async function generateWithWebLlm(cleanWish: string, selectedModel: ModelOption) {
    const engine = await getEngine(selectedModel.id);
    let analysis = createLocalAnalysis(cleanWish);
    let lastReply = "";

    setProgressText("願い事を分解中");

    try {
      const analysisCompletion = await engine.chat.completions.create({
        messages: [
          {
            role: "system",
            content: buildAnalysisPrompt(cleanWish),
          },
          {
            role: "user",
            content: "願い事を分析し、JSONオブジェクトのみを出力してください。",
          },
        ],
        temperature: 0.1,
        max_tokens: 220,
        response_format: {
          type: "json_object",
          schema: ANALYSIS_SCHEMA,
        },
        extra_body: selectedModel.id.startsWith("Qwen3")
          ? { enable_thinking: false }
          : undefined,
      });

      analysis = parseAnalysis(analysisCompletion.choices[0]?.message?.content ?? "");
    } catch (error) {
      console.warn("Using local wish analysis after invalid model output:", error);
    }

    setProgressText("辛口の一撃を生成中");

    for (let attempt = 0; attempt < 2; attempt += 1) {
      const completion = await engine.chat.completions.create({
        messages: [
          {
            role: "system",
            content: buildMajiResPrompt(cleanWish, analysis),
          },
          {
            role: "user",
            content:
              attempt === 0
                ? "分析結果を使って、最高の辛口マジレスを1つだけ作ってください。"
                : "前回は形式崩れか品質不足でした。分析結果に沿う新しい1要素の文字列JSON配列だけを出力してください。",
          },
        ],
        temperature: attempt === 0 ? 0.35 : 0.15,
        max_tokens: 120,
        response_format: {
          type: "json_object",
          schema: MAJI_RES_SCHEMA,
        },
        extra_body: selectedModel.id.startsWith("Qwen3")
          ? { enable_thinking: false }
          : undefined,
      });

      lastReply = completion.choices[0]?.message?.content ?? "";

      try {
        return parseMajiRes(lastReply, cleanWish);
      } catch {
        continue;
      }
    }

    console.warn("Using fallback maji-res after invalid model output:", lastReply);
    return [createFallbackMajiRes(cleanWish, analysis)];
  }

  async function generateMajiRes() {
    const cleanWish = wishText.trim();

    if (!cleanWish) {
      setErrorMessage("願い事を1つ書いてください。空欄にマジレスはできません。");
      setMode("error");
      return;
    }

    if (selectedModel.backend === "web-llm" && !("gpu" in navigator)) {
      setErrorMessage("このブラウザはWebGPUに未対応です。ChromeやEdgeの最新版で試してください。");
      setMode("error");
      return;
    }

    setMode("loading");
    setErrorMessage("");
    setProgress(0);
    setProgressText(`${selectedModel.name}を読み込み中`);
    setMajiResReplies([]);

    try {
      const replies =
        selectedModel.backend === "gemini-nano"
          ? await generateWithGeminiNano(cleanWish)
          : await generateWithWebLlm(cleanWish, selectedModel);

      setMajiResReplies(replies);
      setMode("result");
    } catch (error) {
      console.error(error);
      setErrorMessage(error instanceof Error ? error.message : "短冊AIの返事が少し乱れました。");
      setMode("error");
    }
  }

  function buildShareText() {
    const reply = majiResReplies[0] ?? "";

    return [
      "【マジレス短冊AI】",
      `願い事: ${wishText}`,
      `マジレス: ${reply}`,
      "",
      window.location.href,
    ].join("\n");
  }

  async function copyResult() {
    try {
      await navigator.clipboard.writeText(buildShareText());
      setShareStatus("コピーしました");
    } catch (error) {
      console.error(error);
      setShareStatus("コピーに失敗しました");
    }
  }

  async function shareResult() {
    const text = buildShareText();

    try {
      if ("share" in navigator) {
        await navigator.share({
          title: "マジレス短冊AI",
          text,
          url: window.location.href,
        });
        setShareStatus("共有しました");
        return;
      }

      await copyResult();
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return;
      }

      console.error(error);
      await copyResult();
    }
  }

  function resetForm() {
    setMode("input");
    setErrorMessage("");
    setMajiResReplies([]);
    setShareStatus("");
  }

  return (
    <main className="relative min-h-screen overflow-hidden bg-[#07111f] text-slate-100">
      <div className="star-field absolute inset-0 opacity-80" />
      <div className="absolute inset-x-0 top-0 h-64 bg-gradient-to-b from-[#172348] via-[#132143]/70 to-transparent" />
      <div className="bamboo-stalk absolute -left-14 bottom-0 top-0 w-48 rotate-6 opacity-50" />
      <div className="bamboo-stalk absolute -right-20 bottom-0 top-10 w-56 -rotate-6 opacity-45" />
      <div className="absolute left-1/2 top-10 h-28 w-28 -translate-x-1/2 rounded-full bg-[#f9e7a6] opacity-90 shadow-[0_0_70px_rgba(249,231,166,0.58)]" />

      <section className="relative z-10 mx-auto flex min-h-screen w-full max-w-6xl flex-col px-5 py-8 sm:px-8 lg:px-10">
        <header className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="grid h-11 w-11 place-items-center rounded-full border border-emerald-200/30 bg-emerald-300/10">
              <MoonStar className="h-5 w-5 text-amber-100" />
            </div>
            <div>
              <p className="text-sm text-emerald-100/80">七夕限定・ブラウザ内WASM AI</p>
              <h1 className="text-xl font-semibold tracking-normal text-white sm:text-2xl">
                マジレス短冊AI
              </h1>
            </div>
          </div>
          <div className="hidden items-center gap-2 rounded-full border border-white/15 bg-white/8 px-4 py-2 text-sm text-slate-200 backdrop-blur sm:flex">
            <Sparkles className="h-4 w-4 text-amber-100" />
            {selectedModel.name} Local
          </div>
        </header>

        <div className="grid flex-1 place-items-center py-10">
          {mode === "input" && (
            <section className="grid w-full items-center gap-8 lg:grid-cols-[0.95fr_1.05fr]">
              <div className="max-w-xl">
                <p className="mb-4 inline-flex items-center gap-2 rounded-full border border-amber-100/20 bg-amber-100/10 px-4 py-2 text-sm text-amber-50">
                  <Wand2 className="h-4 w-4" />
                  綺麗な願いを、実行計画に変える夜
                </p>
                <h2 className="text-4xl font-bold leading-tight tracking-normal text-white sm:text-5xl">
                  願い事に、
                  <span className="block text-rose-200">遠慮なく現実を。</span>
                </h2>
                <p className="mt-5 max-w-lg text-base leading-8 text-slate-200">
                  短冊に願いを書いたら、AIが辛口マジレスを一撃だけ返します。
                  ダウンロードも推論もブラウザ内で完結します。
                </p>
                <div className="mt-7 flex flex-wrap gap-2">
                  {sampleWishes.map((sample) => (
                    <button
                      key={sample}
                      type="button"
                      onClick={() => setWishText(sample)}
                      className="rounded-full border border-white/15 bg-white/10 px-3 py-2 text-sm text-slate-100 transition hover:bg-white/15"
                    >
                      {sample}
                    </button>
                  ))}
                </div>
              </div>

              <div className="mx-auto w-full max-w-lg">
                <div className="tanzaku-paper relative rounded-t-sm px-5 pb-8 pt-10 text-slate-950 shadow-tanzaku sm:px-8">
                  <div className="absolute left-1/2 top-4 h-4 w-4 -translate-x-1/2 rounded-full border border-rose-300 bg-rose-100 shadow-inner" />
                  <div className="mx-auto mb-6 h-12 w-px bg-rose-300/80" />
                  <label htmlFor="wish" className="mb-3 block text-sm font-semibold text-rose-900">
                    願い事
                  </label>
                  <textarea
                    id="wish"
                    value={wishText}
                    onChange={(event) => setWishText(event.target.value)}
                    placeholder="例: 来年までに自分のサービスを公開したい"
                    maxLength={120}
                    rows={7}
                    className="min-h-44 w-full resize-none rounded border border-amber-900/20 bg-white/70 px-4 py-4 text-lg leading-8 text-slate-950 outline-none transition placeholder:text-slate-500 focus:border-rose-500 focus:ring-4 focus:ring-rose-200"
                  />
                  <div className="mt-3 flex items-center justify-between gap-4 text-sm text-slate-600">
                    <span>{wishText.length}/120</span>
                    <span>逃げ道はAIが塞ぎます</span>
                  </div>
                  <label htmlFor="model" className="mt-5 block text-sm font-semibold text-rose-900">
                    モデル
                  </label>
                  <div className="mt-2 flex items-start gap-3 rounded border border-amber-900/20 bg-white/60 p-3">
                    <Cpu className="mt-2 h-5 w-5 shrink-0 text-rose-800" />
                    <div className="min-w-0 flex-1">
                      <select
                        id="model"
                        value={selectedModelId}
                        onChange={(event) => setSelectedModelId(event.target.value)}
                        className="w-full rounded border border-amber-900/20 bg-white px-3 py-2 text-sm font-semibold text-slate-950 outline-none focus:border-rose-500 focus:ring-4 focus:ring-rose-200"
                      >
                        {MODEL_OPTIONS.map((model) => (
                          <option key={model.id} value={model.id}>
                            {model.badge}: {model.name} / {model.vram}
                          </option>
                        ))}
                      </select>
                      <p className="mt-2 text-sm leading-6 text-slate-600">{selectedModel.note}</p>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={generateMajiRes}
                    className="mt-6 flex w-full items-center justify-center gap-2 rounded bg-rose-700 px-5 py-4 text-base font-semibold text-white shadow-lg shadow-rose-950/25 transition hover:bg-rose-600 disabled:cursor-not-allowed disabled:bg-slate-400"
                    disabled={!wishText.trim()}
                  >
                    <Send className="h-5 w-5" />
                    AIにマジレスしてもらう
                  </button>
                </div>
              </div>
            </section>
          )}

          {mode === "loading" && (
            <section className="w-full max-w-2xl text-center">
              <div className="mx-auto grid h-20 w-20 place-items-center rounded-full border border-amber-100/30 bg-amber-100/10">
                <Loader2 className="h-9 w-9 animate-spin text-amber-100" />
              </div>
              <h2 className="mt-8 text-3xl font-bold tracking-normal text-white">
                ブラウザ内にAIを召喚中
              </h2>
                  <p className="mt-3 text-slate-200">
                初回はモデルのダウンロードが入ります。軽量モデルなら待ち時間を抑えられます。
              </p>
              <div className="mt-8 rounded border border-white/15 bg-white/10 p-4 backdrop-blur">
                <div className="mb-3 flex items-center justify-between text-sm text-slate-200">
                  <span className="truncate pr-4">{progressText}</span>
                  <span className="font-semibold text-amber-100">{progressPercent}%</span>
                </div>
                <div className="h-3 overflow-hidden rounded-full bg-slate-950/60">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-emerald-300 via-amber-200 to-rose-300 transition-all duration-500"
                    style={{ width: `${progressPercent}%` }}
                  />
                </div>
              </div>
            </section>
          )}

          {mode === "result" && (
            <section className="w-full">
              <div className="mb-7 flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
                <div>
                  <p className="text-sm text-emerald-100/80">願い事: {wishText}</p>
                  <h2 className="mt-2 text-3xl font-bold tracking-normal text-white">
                    マジレス短冊
                  </h2>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={shareResult}
                    className="inline-flex items-center justify-center gap-2 rounded bg-rose-700 px-4 py-3 text-sm font-semibold text-white transition hover:bg-rose-600"
                  >
                    <Share2 className="h-4 w-4" />
                    共有
                  </button>
                  <button
                    type="button"
                    onClick={copyResult}
                    className="inline-flex items-center justify-center gap-2 rounded border border-white/15 bg-white/10 px-4 py-3 text-sm font-semibold text-white transition hover:bg-white/15"
                  >
                    <Copy className="h-4 w-4" />
                    コピー
                  </button>
                  <button
                    type="button"
                    onClick={resetForm}
                    className="inline-flex items-center justify-center gap-2 rounded border border-white/15 bg-white/10 px-4 py-3 text-sm font-semibold text-white transition hover:bg-white/15"
                  >
                    <RefreshCw className="h-4 w-4" />
                    別の願いを書く
                  </button>
                </div>
              </div>

              <div className="mx-auto grid max-w-xl gap-4">
                {majiResReplies.map((reply, index) => (
                  <article
                    key={`${reply}-${index}`}
                    className="tanzaku-paper relative flex min-h-80 flex-col justify-center rounded-t-sm p-8 text-slate-950 shadow-tanzaku"
                  >
                    <div className="absolute left-1/2 top-3 h-3 w-3 -translate-x-1/2 rounded-full border border-rose-300 bg-rose-100" />
                    <p className="text-center text-2xl font-black leading-10 text-rose-900 sm:text-3xl sm:leading-[3rem]">
                      {reply}
                    </p>
                  </article>
                ))}
                {shareStatus && (
                  <p className="text-center text-sm font-semibold text-emerald-100">
                    {shareStatus}
                  </p>
                )}
              </div>
            </section>
          )}

          {mode === "error" && (
            <section className="w-full max-w-xl rounded border border-rose-200/25 bg-rose-950/35 p-6 text-center backdrop-blur">
              <div className="mx-auto grid h-14 w-14 place-items-center rounded-full bg-rose-200/15">
                <Check className="h-7 w-7 text-rose-100" />
              </div>
              <h2 className="mt-5 text-2xl font-bold tracking-normal text-white">
                まだ短冊が整っていません
              </h2>
              <p className="mt-3 leading-7 text-rose-50">{errorMessage}</p>
              <button
                type="button"
                onClick={resetForm}
                className="mt-6 inline-flex items-center justify-center gap-2 rounded bg-white px-5 py-3 text-sm font-semibold text-rose-950 transition hover:bg-rose-50"
              >
                <RefreshCw className="h-4 w-4" />
                入力に戻る
              </button>
            </section>
          )}
        </div>
      </section>
    </main>
  );
}
