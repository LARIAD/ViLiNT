function createEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) {
    el.className = className;
  }
  if (text) {
    el.textContent = text;
  }
  return el;
}

function normalizePath(path) {
  return typeof path === "string" ? path.trim() : "";
}

function setText(id, value, fallback = "") {
  const el = document.getElementById(id);
  if (!el) {
    return;
  }
  el.textContent = value || fallback;
}

function setMeta(id, value, fallback = "") {
  const el = document.getElementById(id);
  if (!el) {
    return;
  }
  el.setAttribute("content", value || fallback);
}

function normalizeTextContent(value) {
  if (Array.isArray(value)) {
    return value
      .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
      .filter(Boolean)
      .join(" ");
  }

  return typeof value === "string" ? value : "";
}

function configureSilentVideo(video) {
  video.controls = true;
  video.autoplay = true;
  video.defaultMuted = true;
  video.muted = true;
  video.volume = 0;
  video.loop = true;
  video.playsInline = true;
  video.addEventListener("volumechange", () => {
    if (!video.muted || video.volume !== 0) {
      video.muted = true;
      video.volume = 0;
    }
  });
}

function iconClassForLink(label) {
  const key = (label || "").toLowerCase();

  if (key.includes("arxiv")) {
    return "ai ai-arxiv";
  }
  if (key.includes("paper") || key.includes("pdf")) {
    return "fas fa-file-pdf";
  }
  if (key.includes("code") || key.includes("github")) {
    return "fab fa-github";
  }
  if (key.includes("video")) {
    return "fas fa-play-circle";
  }
  if (key.includes("project")) {
    return "fas fa-globe";
  }
  if (key.includes("supplement")) {
    return "fas fa-folder-open";
  }

  return "fas fa-link";
}

function normalizeAuthorAffiliations(author) {
  const refs = author.affiliations || [];
  const values = Array.isArray(refs) ? refs : [refs];

  return values.map((ref) => String(ref).trim()).filter(Boolean);
}

function appendAffiliationRefs(parent, refs) {
  if (!refs.length) {
    return;
  }

  const sup = createEl("sup", "affiliation-ref", refs.join(","));
  parent.append(sup);
}

function renderAuthors(authors) {
  const root = document.getElementById("authors");
  root.replaceChildren();

  authors.forEach((author, index) => {
    const wrapper = createEl("span", "author-block");
    const affiliations = normalizeAuthorAffiliations(author);
    let inner;

    if (normalizePath(author.url) && author.url !== "#") {
      inner = document.createElement("a");
      inner.href = author.url;
      inner.textContent = author.name || "Author Name";
    } else {
      inner = document.createElement("span");
      inner.textContent = author.name || "Author Name";
    }

    wrapper.append(inner);
    appendAffiliationRefs(wrapper, affiliations);

    if (index < authors.length - 1) {
      wrapper.append(document.createTextNode(","));
    }

    root.append(wrapper);
  });
}

function renderAffiliations(project, affiliations) {
  const root = document.getElementById("institution");
  if (!root) {
    return;
  }

  root.replaceChildren();

  if (!Array.isArray(affiliations) || affiliations.length === 0) {
    root.textContent = project.institution || "Institution";
    return;
  }

  affiliations.forEach((affiliation, index) => {
    const label = affiliation.name || affiliation.label || "";
    const id = affiliation.id || String(index + 1);

    if (index > 0) {
      root.append(document.createTextNode(" · "));
    }

    const sup = createEl("sup", "affiliation-label", id);
    root.append(sup, document.createTextNode(label));
  });
}

function renderLinks(links) {
  const root = document.getElementById("links");
  root.replaceChildren();

  links.forEach((item) => {
    const span = createEl("span", "link-block");
    const anchor = createEl(
      "a",
      "external-link button is-normal is-rounded is-dark",
      ""
    );

    anchor.href = normalizePath(item.url) || "#";
    if (item.newTab !== false) {
      anchor.target = "_blank";
      anchor.rel = "noreferrer";
    }

    const iconWrap = createEl("span", "icon");
    const icon = createEl("i", iconClassForLink(item.label));
    const label = createEl("span", "", item.label || "Link");

    iconWrap.append(icon);
    anchor.append(iconWrap, label);
    span.append(anchor);
    root.append(span);
  });
}

function renderHighlights(items) {
  const root = document.getElementById("highlights");
  if (!root) {
    return;
  }
  root.replaceChildren();

  items.forEach((item) => {
    const card = createEl("article", "highlight-card");
    const title = createEl("h3", "title is-5", item.title || "Highlight");
    const text = createEl("p", "", item.text || "");
    card.append(title, text);
    root.append(card);
  });
}

function renderVideo(video) {
  const root = document.getElementById("video-shell");
  root.replaceChildren();

  const embedUrl = normalizePath(video.embedUrl);
  const file = normalizePath(video.file);
  const image = normalizePath(video.image);

  if (embedUrl) {
    const iframe = document.createElement("iframe");
    iframe.src = embedUrl;
    iframe.title = "Project video";
    iframe.allow =
      "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture";
    iframe.allowFullscreen = true;
    root.append(iframe);
  } else if (file) {
    const player = document.createElement("video");
    configureSilentVideo(player);
    player.src = file;
    const poster = normalizePath(video.poster);
    if (poster) {
      player.poster = poster;
    }
    root.append(player);
  } else if (image) {
    const img = document.createElement("img");
    img.src = image;
    img.alt = video.alt || "Project front page figure";
    root.append(img);
  } else {
    root.append(
      createEl(
        "div",
        "placeholder",
        "Link a YouTube/Vimeo embed or add a local MP4 in `static/videos/`."
      )
    );
  }

  setText("video-caption", video.caption, "");
}

function renderGallery(items) {
  const section = document.getElementById("results-section");
  const root = document.getElementById("gallery");
  const controls = document.getElementById("results-controls");
  const dotsRoot = document.getElementById("results-dots");
  const prevButton = document.getElementById("results-prev");
  const nextButton = document.getElementById("results-next");
  root.replaceChildren();

  if (!items.length) {
    section.hidden = true;
    return;
  }

  section.hidden = false;
  let index = 0;
  const dots = [];
  const viewport = createEl("div", "results-track-viewport");
  const track = createEl("div", "results-track");
  viewport.append(track);
  root.append(viewport);

  function createGalleryCard(item, role) {
    const isActive = role === "active";
    const article = createEl("article", `results-card is-${role}`);
    let media;

    if (item.type === "video") {
      media = document.createElement("video");
      configureSilentVideo(media);
      if (!isActive) {
        media.controls = false;
      }
    } else {
      media = document.createElement("img");
      media.alt = item.alt || item.caption || "Project result";
    }

    media.src = normalizePath(item.src);
    if (!isActive) {
      media.tabIndex = -1;
    }

    const cardParts = [media];
    if (item.caption) {
      const body = createEl("div", "results-card-body");
      body.append(createEl("p", "", item.caption));
      cardParts.push(body);
    }

    article.append(...cardParts);
    if (!isActive) {
      article.setAttribute("aria-hidden", "true");
    }

    return article;
  }

  function renderItem() {
    track.replaceChildren();

    if (items.length === 1) {
      track.append(createGalleryCard(items[index], "active"));
    } else {
      const previous = items[(index - 1 + items.length) % items.length];
      const current = items[index];
      const next = items[(index + 1) % items.length];

      track.append(
        createGalleryCard(previous, "previous"),
        createGalleryCard(current, "active"),
        createGalleryCard(next, "next")
      );
    }

    dots.forEach((dot, dotIndex) => {
      dot.classList.toggle("is-active", dotIndex === index);
      dot.setAttribute(
        "aria-label",
        dotIndex === index
          ? `Video ${dotIndex + 1} of ${items.length}, current`
          : `Go to video ${dotIndex + 1} of ${items.length}`
      );
    });
  }

  if (controls) {
    controls.hidden = items.length < 2;
  }

  if (dotsRoot) {
    dotsRoot.replaceChildren();
    items.forEach((_, dotIndex) => {
      const dot = createEl("button", "results-dot");
      dot.type = "button";
      dot.onclick = () => {
        index = dotIndex;
        renderItem();
      };
      dots.push(dot);
      dotsRoot.append(dot);
    });
  }

  if (prevButton) {
    prevButton.onclick = () => {
      index = (index - 1 + items.length) % items.length;
      renderItem();
    };
  }

  if (nextButton) {
    nextButton.onclick = () => {
      index = (index + 1) % items.length;
      renderItem();
    };
  }

  renderItem();
}

function setupNavbarBurger() {
  const burgers = Array.from(document.querySelectorAll(".navbar-burger"));
  burgers.forEach((burger) => {
    burger.addEventListener("click", () => {
      const target = burger.dataset.target;
      const menu = document.getElementById(target);
      burger.classList.toggle("is-active");
      if (menu) {
        menu.classList.toggle("is-active");
      }
    });
  });
}

async function bootstrap() {
  const response = await fetch("content/site.json");
  const site = await response.json();

  document.title = site.project?.title || "Project Page";
  setMeta(
    "meta-description",
    site.meta?.description,
    "Research project page"
  );
  setMeta(
    "meta-keywords",
    site.meta?.keywords,
    "research, project page"
  );
  setText("brand-title", site.nav?.brand || "Project Page");
  setText("project-title", site.project?.title, "Project Title");
  setText("project-subtitle", site.project?.subtitle, "Project subtitle");
  setText("abstract", normalizeTextContent(site.abstract), "");
  setText(
    "navigation-transformer",
    normalizeTextContent(site.navigationTransformer),
    ""
  );
  setText("bibtex", site.citation?.bibtex, "");
  setText(
    "footer-note",
    site.footerNote,
    "This webpage template is inspired by Nerfies-style academic project pages."
  );

  const homeLink = document.getElementById("home-link");
  if (homeLink) {
    homeLink.href = normalizePath(site.nav?.homeUrl) || "#top";
  }

  renderAuthors(site.authors || []);
  renderAffiliations(site.project || {}, site.affiliations || []);
  renderLinks(site.links || []);
  renderVideo(site.video || {});
  renderGallery(site.gallery || []);
  setupNavbarBurger();
}

bootstrap().catch((error) => {
  console.error("Failed to load project page content.", error);
});
