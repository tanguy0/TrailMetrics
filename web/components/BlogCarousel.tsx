"use client";

/**
 * Swipes through an article's rasterized PDF pages.
 *
 * Plain state and a handful of event listeners — a carousel this simple does not
 * earn a new npm dependency. Each page is a big PNG, so only the current one, plus
 * its immediate neighbours, ever render an `<img>`; the rest are placeholders that
 * mount as the reader approaches them.
 */

import { useCallback, useEffect, useState } from "react";

export function BlogCarousel({ pageUrls, title }: { pageUrls: string[]; title: string }) {
  const [index, setIndex] = useState(0);
  const count = pageUrls.length;

  const go = useCallback(
    (delta: number) => {
      setIndex((current) => Math.min(Math.max(current + delta, 0), count - 1));
    },
    [count],
  );

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "ArrowRight") go(1);
      if (event.key === "ArrowLeft") go(-1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go]);

  if (count === 0) return null;

  let touchStartX: number | null = null;
  const onTouchStart = (event: React.TouchEvent) => {
    touchStartX = event.touches[0]?.clientX ?? null;
  };
  const onTouchEnd = (event: React.TouchEvent) => {
    if (touchStartX === null) return;
    const endX = event.changedTouches[0]?.clientX ?? touchStartX;
    const delta = touchStartX - endX;
    // Ignore small drags — a tap or a scroll attempt should not flip the page.
    if (Math.abs(delta) > 40) go(delta > 0 ? 1 : -1);
    touchStartX = null;
  };

  return (
    <div className="blog-carousel">
      <div
        className="blog-carousel__viewport"
        onTouchStart={onTouchStart}
        onTouchEnd={onTouchEnd}
      >
        {pageUrls.map((url, i) => (
          <img
            key={url}
            className="blog-carousel__slide"
            style={{ display: i === index ? "block" : "none" }}
            src={url}
            alt={`${title} — page ${i + 1} / ${count}`}
            loading={Math.abs(i - index) <= 1 ? "eager" : "lazy"}
          />
        ))}

        {index > 0 && (
          <button
            type="button"
            className="blog-carousel__arrow blog-carousel__arrow--prev"
            onClick={() => go(-1)}
            aria-label="Previous page"
          >
            ‹
          </button>
        )}
        {index < count - 1 && (
          <button
            type="button"
            className="blog-carousel__arrow blog-carousel__arrow--next"
            onClick={() => go(1)}
            aria-label="Next page"
          >
            ›
          </button>
        )}
      </div>

      {count > 1 && (
        <div className="blog-carousel__dots">
          {pageUrls.map((url, i) => (
            <button
              key={url}
              type="button"
              className={
                "blog-carousel__dot" + (i === index ? " blog-carousel__dot--active" : "")
              }
              onClick={() => setIndex(i)}
              aria-label={`Go to page ${i + 1}`}
            />
          ))}
        </div>
      )}
    </div>
  );
}
