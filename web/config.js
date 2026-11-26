/**
 * Voice Changer Web Configuration
 * Manages model base URL (Local / Hugging Face Hub / Custom CDN)
 */
(function () {
  "use strict";

  // Change this to your Hugging Face model repository when publishing,
  // e.g. "your-username/vc-web-models"
  // If set to null, it will default to local "/models" unless specified in URL params.
  window.__VC_DEFAULT_HF_REPO__ = window.__VC_DEFAULT_HF_REPO__ || "sakurailina/vc-web-models";

  const params = new URLSearchParams(typeof location !== "undefined" ? location.search : "");
  const customBase = params.get("model_base");
  const customHf = params.get("hf_repo");

  const isLocalHost =
    typeof location !== "undefined" &&
    (location.hostname === "localhost" || location.hostname === "127.0.0.1" || location.hostname === "[::1]");

  let base = "/models";
  if (customBase) {
    base = customBase.replace(/\/+$/, "");
  } else if (customHf) {
    base = `https://huggingface.co/${customHf.replace(/^\/+|\/+$/g, "")}/resolve/main/models`;
  } else if (!isLocalHost && window.__VC_DEFAULT_HF_REPO__) {
    base = `https://huggingface.co/${window.__VC_DEFAULT_HF_REPO__.replace(/^\/+|\/+$/g, "")}/resolve/main/models`;
  }

  window.VC_CONFIG = {
    modelBaseUrl: base,
    getModelUrl: function (subPath) {
      const clean = subPath.replace(/^\/+/, "");
      return `${this.modelBaseUrl}/${clean}`;
    },
    setModelSource: function (source) {
      if (!source || source === "local") {
        this.modelBaseUrl = "/models";
      } else if (source.startsWith("http://") || source.startsWith("https://")) {
        this.modelBaseUrl = source.replace(/\/+$/, "");
      } else {
        // Assume Hugging Face repo format "user/repo"
        this.modelBaseUrl = `https://huggingface.co/${source.replace(/^\/+|\/+$/g, "")}/resolve/main/models`;
      }
      console.log(`[VC-Config] Model base URL updated to: ${this.modelBaseUrl}`);
    },
  };

  console.log(`[VC-Config] Initialized. Model base URL: ${window.VC_CONFIG.modelBaseUrl}`);
})();
