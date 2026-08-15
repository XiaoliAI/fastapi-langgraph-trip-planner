/// <reference types="vite/client" />

declare module '*.css'

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string
  readonly VITE_AMAP_WEB_JS_KEY?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
