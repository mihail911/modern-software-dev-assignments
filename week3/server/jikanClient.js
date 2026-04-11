const BASE_URL = "https://api.jikan.moe/v4";

export class JikanClient {
  async #request(path) {
    const url = `${BASE_URL}${path}`;
    const res = await fetch(url);

    if (res.status === 429) {
      const retryAfter = res.headers.get("Retry-After") ?? "a moment";
      throw new Error(`Jikan rate limit exceeded — please retry after ${retryAfter}s`);
    }
    if (res.status === 404) {
      throw new Error(`Jikan API: 404 Not Found — ${path}`);
    }
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`Jikan API: ${res.status} ${res.statusText} — ${body}`);
    }

    return res.json();
  }

  async searchAnime(q, { limit = 10, type, status, rating } = {}) {
    const params = new URLSearchParams({ q, limit: String(limit) });
    if (type) params.set("type", type);
    if (status) params.set("status", status);
    if (rating) params.set("rating", rating);

    const { data } = await this.#request(`/anime?${params}`);
    return (data ?? []).map(this.#toAnimeSummary);
  }

  async getAnime(id) {
    const { data } = await this.#request(`/anime/${id}`);
    return {
      mal_id: data.mal_id,
      title: data.title,
      title_english: data.title_english ?? null,
      title_japanese: data.title_japanese ?? null,
      type: data.type,
      episodes: data.episodes ?? null,
      status: data.status,
      score: data.score ?? null,
      rank: data.rank ?? null,
      popularity: data.popularity ?? null,
      rating: data.rating ?? null,
      synopsis: data.synopsis ?? null,
      genres: data.genres?.map((g) => g.name) ?? [],
      themes: data.themes?.map((t) => t.name) ?? [],
      studios: data.studios?.map((s) => s.name) ?? [],
      aired: data.aired?.string ?? null,
      url: data.url,
    };
  }

  async getTopAnime({ limit = 10, type } = {}) {
    const params = new URLSearchParams({ limit: String(limit) });
    if (type) params.set("type", type);

    const { data } = await this.#request(`/top/anime?${params}`);
    return (data ?? []).map(this.#toAnimeSummary);
  }

  async searchManga(q, { limit = 10, type, status } = {}) {
    const params = new URLSearchParams({ q, limit: String(limit) });
    if (type) params.set("type", type);
    if (status) params.set("status", status);

    const { data } = await this.#request(`/manga?${params}`);
    return (data ?? []).map((m) => ({
      mal_id: m.mal_id,
      title: m.title,
      title_english: m.title_english ?? null,
      type: m.type,
      chapters: m.chapters ?? null,
      volumes: m.volumes ?? null,
      status: m.status,
      score: m.score ?? null,
      rank: m.rank ?? null,
      genres: m.genres?.map((g) => g.name) ?? [],
      synopsis: m.synopsis ?? null,
      url: m.url,
    }));
  }

  #toAnimeSummary(a) {
    return {
      mal_id: a.mal_id,
      title: a.title,
      title_english: a.title_english ?? null,
      type: a.type,
      episodes: a.episodes ?? null,
      status: a.status,
      score: a.score ?? null,
      rank: a.rank ?? null,
      genres: a.genres?.map((g) => g.name) ?? [],
      synopsis: a.synopsis ?? null,
      url: a.url,
    };
  }
}
