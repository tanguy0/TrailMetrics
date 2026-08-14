/**
 * Fixed sign-off shown at the end of every blog article.
 *
 * Deliberately not a per-post field: every article has the same one author, so
 * there is nothing here for the write form to ask about — see `param.text.body.help`
 * in `src/translations.py` for the app's usual stance on text that's the author's
 * own words rather than translated chrome.
 */

const LINKEDIN_URL = "https://www.linkedin.com/in/tblervacque/";

export function BlogSignature() {
  return (
    <footer className="blog-signature">
      <p>
        Écrit par <strong>Tanguy Blervacque</strong>, coach et data scientist
        spécialisé en course à pied et trail.
      </p>
      <a href={LINKEDIN_URL} target="_blank" rel="noopener noreferrer">
        Retrouvez-moi sur LinkedIn ↗
      </a>
    </footer>
  );
}
