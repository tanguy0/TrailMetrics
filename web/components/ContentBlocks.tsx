"use client";

/**
 * The two pieces of panel content that are not data: prose and images.
 *
 * Their presentation lives here for the same reason `ChartView` owns chart
 * presentation — the plot definition describes *what* the block is (a heading, an
 * image at 60% width) and this decides how it looks, so the styling is defined once
 * however many blocks a page carries.
 */

import type { ImageBlock, TextBlock } from "@/lib/types";

/**
 * A block of the athlete's own text.
 *
 * Rendered as text, never as markup: the content is user input and injecting it as
 * HTML would make every saved page a stored-XSS vector. Line breaks are preserved in
 * CSS (`white-space: pre-wrap`) instead, which is what someone typing a paragraph
 * expects anyway.
 */
export function TextBlockView({ block }: { block: TextBlock }) {
  if (!block.text.trim()) return null;

  const className = [
    "text-block",
    `text-block--${block.variant}`,
    block.align === "center" ? "text-block--center" : "",
    block.tone !== "none" ? `text-block--${block.tone}` : "",
  ]
    .filter(Boolean)
    .join(" ");

  // A heading is a heading in the document outline too, not just visually.
  if (block.variant === "heading") {
    return <h4 className={className}>{block.text}</h4>;
  }
  return <p className={className}>{block.text}</p>;
}

/** An image, at the width and alignment the author chose. */
export function ImageBlockView({ block }: { block: ImageBlock }) {
  if (!block.src.trim()) return null;

  return (
    <figure
      className={`image-block${block.align === "center" ? " image-block--center" : ""}`}
    >
      {/* A plain <img>: the source is either this app's own asset route or an
          arbitrary external URL, and the Next image loader would need every one of
          those hosts allowlisted up front. */}
      <img
        className="image-block__img"
        src={block.src}
        alt={block.alt}
        style={{ width: `${block.width_pct}%` }}
      />
      {block.caption && (
        <figcaption className="image-block__caption">{block.caption}</figcaption>
      )}
    </figure>
  );
}
