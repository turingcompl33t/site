+++
title = '{{ replace .File.ContentBaseName "-" " " | title }}'
date = {{ .Date }}
description = ''
venue = ''
external = ''
tags = []
draft = true

# An article published elsewhere. It appears in the writing list, tag pages,
# and "Recent", linking out to `external`. render = 'link' (not 'never') keeps it
# on tag pages; no page is generated here, so it is also kept out of the sitemap.
[build]
  render = 'link'
  list = 'always'

[sitemap]
  disable = true
+++
