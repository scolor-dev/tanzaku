import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

declare const process: {
  env: Record<string, string | undefined>;
};

const repositoryName = process.env.GITHUB_REPOSITORY?.split("/")[1];

export default defineConfig({
  base: repositoryName ? `/${repositoryName}/` : "/",
  plugins: [react()],
  worker: {
    format: "es",
  },
});
