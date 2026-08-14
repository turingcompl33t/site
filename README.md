## Personal Site

The content for my personal website, built with [Hugo](http://gohugo.io/).

Serve locally with `hugo server -D` (`-D` includes drafts).

### TODO before launch

- [ ] Set the real `baseURL` in `hugo.toml` — currently `https://example.com/`, which puts the wrong domain in every `<link rel="canonical">` and in `sitemap.xml`.
- [ ] Write the homepage bio in `content/_index.md`. Still a stub, and the only place on the site that describes me now that About is gone.
- [ ] Fill in `params.email` / `params.github` in `hugo.toml`. Both empty, so the footer renders nothing but the copyright and there's no contact info anywhere.
- [ ] Decide on `content/essays/2026-02-12-makemore-nn` — unfinished (contains a literal `TODO` and outline notes) but currently published. Probably wants `draft = true`.
- [ ] Localize the hotlinked Cornell diagram used in `micrograd0` and `micrograd1` (`blogs.cornell.edu/info2040/...`) — a 2015 URL on someone else's server, and the last external image dependency.
