// Sélecteurs YouTube pour commentaires
const COMMENT_SELECTOR = "#content-text"; // texte d'un commentaire
const REPLY_SELECTOR = "ytd-comment-replies-renderer #content-text"; // réponses visibles

// Extraire les commentaires visibles (top-level + replies visibles)
function extractVisibleComments() {
  const topLevel = Array.from(document.querySelectorAll(COMMENT_SELECTOR))
    .map(el => el.innerText.trim())
    .filter(t => t.length > 0);

  // Replies visibles (threads)
  const replies = Array.from(document.querySelectorAll(REPLY_SELECTOR))
    .map(el => el.innerText.trim())
    .filter(t => t.length > 0);

  // Déduplique
  const all = [...new Set([...topLevel, ...replies])];

  return all;
}

// Écouter les messages venant du popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "GET_COMMENTS") {
    const comments = extractVisibleComments();
    sendResponse({ comments });
  }
});
