export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type ModelInfo = {
  id: string; name: string; license: string; commercial_use: boolean;
  languages: string[]; features: Record<string, boolean>; available: boolean; unavailable_reason: string;
};
export type Job = {
  id: string; status: "queued" | "running" | "succeeded" | "failed";
  audio_id: string | null; error: string | null; metrics: Record<string, unknown> | null;
};

async function j<T>(res: Response): Promise<T> {
  const body = await res.json();
  if (!res.ok) throw new Error(body?.error?.message ?? `HTTP ${res.status}`);
  return body as T;
}

export const getModels = () => fetch(`${API}/api/v1/models`).then((r) => j<{ models: ModelInfo[] }>(r));
export const generate = (body: object) =>
  fetch(`${API}/api/v1/tts/generate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
    .then((r) => j<{ job_id: string }>(r));
export const getJob = (id: string) => fetch(`${API}/api/v1/jobs/${id}`).then((r) => j<Job>(r));
export const audioUrl = (id: string) => `${API}/api/v1/audio/${id}`;
export type LangMatrix = Record<string, { name: string; models: string[] }>;
export const getLanguages = () => fetch(`${API}/api/v1/languages`).then((r) => j<{ languages: LangMatrix }>(r));
export type Voice = { id: string; name: string; model: string; languages: string[]; description: string; style: string;
  license: string; sample_url: string | null };
export const getVoices = (language: string, model: string) =>
  fetch(`${API}/api/v1/voices?language=${language}&model=${model}`).then((r) => j<{ voices: Voice[] }>(r));
