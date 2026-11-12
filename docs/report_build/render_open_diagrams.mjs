import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { Graphviz } from "@hpcc-js/wasm-graphviz";

const root = dirname(fileURLToPath(import.meta.url));
const outDir = join(root, "figures");
const srcDir = join(root, "diagrams");
const chrome = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
mkdirSync(outDir, { recursive: true });
mkdirSync(srcDir, { recursive: true });

const theme = {
  theme: "base",
  themeVariables: {
    background: "#f4f1ea",
    fontFamily: "Segoe UI, Calibri, sans-serif",
    fontSize: "20px",
    primaryColor: "#e5f2ec",
    primaryTextColor: "#142028",
    primaryBorderColor: "#0f5c4c",
    secondaryColor: "#f8ecd0",
    secondaryTextColor: "#142028",
    secondaryBorderColor: "#8a6220",
    tertiaryColor: "#d9e5f4",
    tertiaryTextColor: "#142028",
    tertiaryBorderColor: "#1d4e89",
    lineColor: "#1d4e89",
    textColor: "#142028",
    mainBkg: "#e5f2ec",
    nodeBorder: "#0f5c4c",
    clusterBkg: "#fffdf8",
    clusterBorder: "#1d4e89",
    titleColor: "#14324a",
    edgeLabelBackground: "#fffdf8",
    actorBkg: "#0f5c4c",
    actorTextColor: "#ffffff",
    actorBorder: "#14342c",
    actorLineColor: "#8aa0b2",
    signalColor: "#142028",
    signalTextColor: "#142028",
    noteBkgColor: "#fff6df",
    noteTextColor: "#142028",
    noteBorderColor: "#8a6220",
    activationBkgColor: "#d9e5f4",
    activationBorderColor: "#1d4e89",
    sequenceNumberColor: "#ffffff",
    labelBoxBkgColor: "#fff6df",
    labelBoxBorderColor: "#8a6220",
    labelTextColor: "#142028",
  },
  flowchart: { htmlLabels: true, curve: "basis", padding: 16, nodeSpacing: 36, rankSpacing: 44, wrappingWidth: 320 },
  sequence: { actorMargin: 28, messageMargin: 36, mirrorActors: false, wrap: true },
};
const themePath = join(srcDir, "theme.json");
writeFileSync(themePath, JSON.stringify(theme, null, 2));

const DOT = `
graph [
  bgcolor="#f4f1ea" pad="0.35" nodesep="0.42" ranksep="0.55"
  fontname="Arial" fontsize=20 labelloc=t labeljust=l
];
node [
  fontname="Arial" fontsize=16 style="filled,rounded" shape=box
  color="#1c3d34" fillcolor="#e7f2ee" penwidth=1.5 margin="0.28,0.14"
];
edge [fontname="Arial" fontsize=13 color="#1d4e89" fontcolor="#142028" arrowsize=0.7];
`;

const graphs = {
  architecture: `digraph {
    ${DOT}
    label="Where work happens\\lIndexing fills the index. A question only reads it.\\l";
    rankdir=TB;
    subgraph cluster_index {
      label="Once per PDF";
      color="#0f5c4c";
      fontcolor="#0f5c4c";
      style="rounded";
      pdf [label="PDF file\\npath in data/raw or data/pdfs"];
      ident [label="SHA-256 document id\\nreject empty or corrupt files"];
      pages [label="pypdf page text\\nOCR only if a page is sparse"];
      chunks [label="Chunks\\n1,200 max, overlap 100, merge under 300"];
      vectors [label="all-MiniLM-L6-v2\\n384 floats, batches of 32"];
      pdf -> ident -> pages -> chunks -> vectors;
    }
    es [shape=cylinder, fillcolor="#d9e5f4", color="#1d4e89", label="rag_corpus_v1\\nElasticsearch 8.12.2\\n12,054 documents"];
    vectors -> es [label="bulk index\\n_id = chunk id"];
    subgraph cluster_ask {
      label="Every question";
      color="#1d4e89";
      fontcolor="#1d4e89";
      style="rounded";
      q [fillcolor="#f8ecd0", color="#8a6220", label="Question text"];
      bm [label="BM25\\npool of 20"];
      knn [label="cosine kNN\\n50 candidates, keep 20"];
      fuse [fillcolor="#f8ecd0", color="#8a6220", label="min-max fusion\\n0.3 keyword + 0.7 semantic\\nkeep 5"];
      gate [shape=diamond, label="score ≥ 0.35?"];
      llm [label="LLM at temperature 0\\nmock by default\\ndemo used Llama 3.2"];
      ans [fillcolor="#14324a", fontcolor="white", color="#14324a", label="RAGAnswer\\nprose + source list"];
      q -> bm;
      q -> knn;
      bm -> fuse;
      knn -> fuse;
      fuse -> gate;
      gate -> llm [label="yes"];
      gate -> ans [label="no: fixed abstain sentence"];
      llm -> ans;
    }
    es -> bm [style=dashed];
    es -> knn [style=dashed];
  }`,
  clean: `digraph {
    ${DOT}
    label="Cleaning is a chain of rewrites, not a model call\\l";
    rankdir=TB;
    raw [label="pypdf string"];
    controls [label="Drop controls\\nkeep tab and newline"];
    hyphen [label="Join a hyphenated\\nline break"];
    space [label="Collapse spaces\\nand blank lines"];
    header [label="After page 1, drop\\npage numbers and\\nshort running heads"];
    blocks [label="Paragraph blocks\\nfor the chunker"];
    raw -> controls -> hyphen -> space -> header -> blocks;
  }`,
  chunk: `digraph {
    ${DOT}
    label="Three passes over the same blocks\\l";
    rankdir=TB;
    subgraph cluster_pack {
      label="1  Pack by heading";
      color="#0f5c4c"; fontcolor="#0f5c4c"; style="rounded";
      s1 [label="Section starts as Introduction"];
      s2 [label="A heading closes the group\\nand becomes the next section"];
      s3 [label="Add blocks until the next one\\nwould pass 1,200 characters"];
      s1 -> s2 -> s3;
    }
    subgraph cluster_cut {
      label="2  Keep a tail, then cut";
      color="#8a6220"; fontcolor="#8a6220"; style="rounded";
      c1 [label="Carry the last 100 characters\\nfrom a word boundary"];
      c2 [label="If still too long, cut on a\\nblank line, newline, or sentence"];
      c1 -> c2;
    }
    subgraph cluster_merge {
      label="3  Reject, then merge";
      color="#1d4e89"; fontcolor="#1d4e89"; style="rounded";
      m1 [label="Drop empties, chunks under 20\\ncharacters, and near-duplicates\\nat similarity 0.92"];
      m2 [label="Merge a chunk under 300 characters\\ninto the same section only\\nceiling 1,560"];
      m1 -> m2;
    }
    s3 -> c1;
    c2 -> m1;
    example [fillcolor="#f8ecd0", color="#8a6220", label="82f38954477f2c53.pdf\\n95 blocks to 61 chunks\\n3 merges, 0 LLM calls"];
    m2 -> example;
  }`,
  bm25: `digraph {
    ${DOT}
    label="BM25 is computed by Lucene. This project does not set k1 or b.\\l";
    rankdir=TB;
    query [fillcolor="#f8ecd0", color="#8a6220", label="Elasticsearch match on content\\noptional figure boost 4.0"];
    tf [label="Term frequency\\nhow often the word is in the chunk\\nextra repeats count less\\nk1 default 1.2"];
    idf [label="Inverse document frequency\\nrare words outweigh the word the"];
    len [label="Length normalisation\\na long chunk does not win\\nonly because it is long\\nb default 0.75"];
    score [fillcolor="#14324a", fontcolor="white", color="#14324a", label="score(chunk, question)\\nsum over the question terms"];
    query -> tf -> idf -> len -> score;
  }`,
  hnsw: `digraph {
    ${DOT}
    label="HNSW inside Elasticsearch\\lindex = true, similarity = cosine\\lm and ef_construction are not set\\lno saved recall or latency\\l";
    rankdir=TB;
    q [shape=ellipse, fillcolor="#f8ecd0", color="#8a6220", label="query vector\\nk and num_candidates = 50"];
    subgraph cluster_top {
      label="Upper layer: a few long-range links";
      color="#8a6220"; fontcolor="#8a6220"; style="dashed";
      { rank=same; t1 [label="entry"]; t2 [label="long hop"]; }
      t1 -> t2 [color="#8a6220", label="greedy", constraint=false];
    }
    subgraph cluster_base {
      label="Base layer: every stored vector";
      color="#1d4e89"; fontcolor="#1d4e89"; style="rounded";
      { rank=same;
        b1 [shape=circle, label="", width=0.42, fixedsize=true];
        b2 [shape=circle, label="", width=0.42, fixedsize=true];
        near [shape=circle, label="near", width=0.78, fixedsize=true, fillcolor="#14324a", fontcolor="white", color="#14324a", fontsize=13];
        b3 [shape=circle, label="", width=0.42, fixedsize=true];
      }
      b1 -> b2 -> near -> b3 [color="#8aa0b2", arrowsize=0.45, constraint=false];
    }
    q -> t1 [label="enter at the top"];
    t1 -> b2 [label="drop a layer", color="#8a6220"];
    t2 -> near [color="#8a6220", label="closer"];
  }`,
  hybrid: `digraph {
    ${DOT}
    label="Two independent lists, then one score\\l";
    rankdir=TB;
    subgraph cluster_kw {
      label="Keyword arm\\nweight 0.3";
      color="#0f5c4c"; fontcolor="#0f5c4c"; style="rounded";
      k1 [label="BM25 match on content"];
      k2 [label="Take 20 hits"];
      k3 [label="Min-max only those 20"];
      k1 -> k2 -> k3;
    }
    subgraph cluster_sem {
      label="Semantic arm\\nweight 0.7";
      color="#1d4e89"; fontcolor="#1d4e89"; style="rounded";
      s1 [label="Embed the question\\nwith the same MiniLM"];
      s2 [label="kNN, 50 candidates\\nkeep 20"];
      s3 [label="Min-max only those 20"];
      s1 -> s2 -> s3;
    }
    miss [fillcolor="#fff6df", color="#8a6220", label="A chunk missing from one arm\\nscores 0 on that arm"];
    fuse [fillcolor="#14324a", fontcolor="white", color="#14324a", label="0.3 × norm(BM25)\\n+ 0.7 × norm(cosine)"];
    boost [label="Then, if they apply\\nfigure or table match  +4.0\\nvisual image chunk  +2.0"];
    top [fillcolor="#f8ecd0", color="#8a6220", label="Sort by score, then chunk id\\nkeep 5"];
    rrf [shape=box, fillcolor="#fff6df", color="#8a6220", label="RRF is coded as 1 / (60 + rank).\\nIt runs only if fusion_strategy is rrf.\\nThe active setting is min_max."];
    k3 -> s1 [style=invis, weight=8];
    k3 -> fuse;
    s3 -> fuse;
    miss -> fuse [style=dashed];
    fuse -> boost -> top;
    top -> rrf [style=dotted, arrowhead=none];
  }`,
  ocr: `digraph {
    ${DOT}
    label="OCR runs only when a page is sparse\\l";
    rankdir=TB;
    count [shape=diamond, label="Fewer than 15\\nnon-space characters?"];
    skip [label="Keep the text page\\nTesseract is not called"];
    mu [shape=diamond, label="PyMuPDF\\nopened the file?"];
    bin [shape=diamond, label="Tesseract\\nbinary runs?"];
    render [label="Render at 300 DPI\\nlanguage eng"];
    useful [shape=diamond, label="OCR text still\\nunder 15 characters?"];
    image [fillcolor="#f8ecd0", color="#8a6220", label="IMAGE_ONLY\\nprocessing continues"];
    keep [fillcolor="#14324a", fontcolor="white", color="#14324a", label="Replace the page text\\nstatus SUCCESS"];
    host [fillcolor="#fff6df", color="#8a6220", label="Test host\\nbinary was not on PATH\\n31 stored pages stay flagged\\nDocker installs tesseract-ocr"];
    count -> skip [label="no"];
    count -> mu [label="yes"];
    mu -> image [label="no"];
    mu -> bin [label="yes"];
    bin -> image [label="no"];
    bin -> render [label="yes"];
    render -> useful;
    useful -> image [label="yes"];
    useful -> keep [label="no"];
    image -> host;
  }`,
  context: `digraph {
    ${DOT}
    label="After fusion, on the ask path\\lThe gate runs before any prompt is built\\l";
    rankdir=TB;
    subgraph cluster_rank {
      label="Rank";
      color="#0f5c4c"; fontcolor="#0f5c4c"; style="rounded";
      r1 [label="Keep the fused order"];
      r2 [label="There is no cross-encoder"];
      r1 -> r2;
    }
    subgraph cluster_gate {
      label="Gate";
      color="#8a6220"; fontcolor="#8a6220"; style="rounded";
      g1 [fillcolor="#f8ecd0", color="#8a6220", label="No hits: abstain"];
      g2 [fillcolor="#f8ecd0", color="#8a6220", label="First score below 0.35: abstain"];
      g3 [fillcolor="#f8ecd0", color="#8a6220", label="A requested role is missing: abstain"];
      g1 -> g2 -> g3 [style=invis];
    }
    subgraph cluster_prompt {
      label="Prompt";
      color="#1d4e89"; fontcolor="#1d4e89"; style="rounded";
      p1 [label="Format document, pages, section,\\nchunk id, and the text"];
      p2 [label="Concatenate every hit\\nthere is no token budget"];
      p3 [label="The stored demo then added\\n5 knowledge-graph rows"];
      p1 -> p2 -> p3;
    }
    r2 -> g1;
    g3 -> p1 [label="only if the gate passes"];
  }`,
  prompt: `digraph {
    ${DOT}
    label="The ask path sends two messages\\l";
    rankdir=TB;
    sys [fillcolor="#14324a", fontcolor="white", color="#14324a", label="System\\l\\lUse only the retrieved context.\\lKeep technical terms.\\lIf it is not enough,\\lanswer exactly:\\lThe information is not available\\lin the provided context.\\lName the PDF and its pages."];
    user [fillcolor="#e7f2ee", label="User\\l\\lRetrieved Unified Context\\lthen each hit block\\l\\lUser Question\\lthen the query\\l\\lAnswer only from the context\\land cite its provenance."];
    call [fillcolor="#f8ecd0", color="#8a6220", label="POST messages\\ntemperature 0\\nno max_tokens field"];
    out [label="Completion text\\nor the mock sentence\\nif provider is mock"];
    sys -> user [style=invis];
    sys -> call;
    user -> call;
    call -> out;
  }`,
  citation: `digraph {
    ${DOT}
    label="Question 5 keeps two citation layers in one JSON file\\l";
    rankdir=TB;
    hit [label="Retrieved chunk_002\\n82f38954477f2c53.pdf\\npage 1\\nhybrid score 0.6678"];
    app [fillcolor="#14324a", fontcolor="white", color="#14324a", label="Application source list\\nbuild_source_references\\npage number comes from the chunker"];
    model [fillcolor="#f8ecd0", color="#8a6220", label="Sentences in the answer\\nLlama 3.2 names a different chunk\\nand then infers"];
    rule [shape=box, fillcolor="#fff6df", color="#8a6220", label="When they disagree,\\nthe sources array is the\\ncitation the program attached."];
    sum [label="The summary of chunk_002\\nis also a hit\\npage list is empty"];
    hit -> app;
    sum -> app [style=dashed];
    hit -> model [style=dashed, label="evidence in the prompt"];
    model -> rule [style=dotted];
    app -> rule;
  }`,
  minmax: `digraph {
    graph [bgcolor="#f4f1ea" pad="0.45" fontname="Arial" fontsize=18 labelloc=t labeljust=l];
    label="Invented scores, to show the arithmetic.\\lThe demo JSON stores only the fused number.\\l";
    node [shape=none fontname="Arial" fontsize=16];
    board [label=<
      <table border="0" cellborder="1" cellspacing="0" cellpadding="10" color="#1c3d34">
        <tr>
          <td bgcolor="#14324a"><font color="white" face="Arial"><b>Chunk</b></font></td>
          <td bgcolor="#14324a"><font color="white" face="Arial"><b>BM25</b></font></td>
          <td bgcolor="#14324a"><font color="white" face="Arial"><b>Norm  </b></font></td>
          <td bgcolor="#14324a"><font color="white" face="Arial"><b>Vector</b></font></td>
          <td bgcolor="#14324a"><font color="white" face="Arial"><b>Norm  </b></font></td>
          <td bgcolor="#8a6220"><font color="white" face="Arial"><b>0.3 + 0.7</b></font></td>
        </tr>
        <tr>
          <td bgcolor="#e7f2ee">A</td><td>12</td><td>1</td><td>0.40</td><td>0</td><td bgcolor="#f8ecd0">0.30</td>
        </tr>
        <tr>
          <td bgcolor="#e7f2ee">B</td><td>6</td><td>0</td><td>0.90</td><td>1</td><td bgcolor="#0f5c4c"><font color="white" face="Arial">0.70 first</font></td>
        </tr>
        <tr>
          <td bgcolor="#e7f2ee">C</td><td>9</td><td>0.5</td><td>0.65</td><td>0.5</td><td bgcolor="#f8ecd0">0.50</td>
        </tr>
      </table>
    >];
  }`,
};

const mermaid = {
  pipeline: `timeline
    title One PDF is indexed once. Each question only searches.
    section Index time
        Identify : SHA-256 of the file bytes
        Extract : pypdf text, blocks, OCR flag
        Chunk : 1200 max, overlap 100, same-section merge
        Embed : MiniLM, 384 dimensions, batch 32
        Store : bulk index, id is the chunk id
    section Question time
        Search : BM25 pool 20 and kNN pool 20
        Fuse : min-max, weights 0.3 and 0.7
        Keep : top 5
        Gate : abstain if the first score is below 0.35
        Write : temperature 0, plus the source list
`,
  ingest: `flowchart TD
    start(["Three ways in"])
    start --> one["ingest --pdf\\ncopy one file into data/raw"]
    start --> many["ingest-all\\nevery PDF in data/raw"]
    start --> crawled["process-pdfs\\nmanifest in data/pdfs"]
    one --> proc["process_pdf"]
    many --> proc
    crawled --> proc
    proc --> ok{"Bytes open as a PDF\\nwith at least one page?"}
    ok -->|no| bad["That file stops\\nFileNotFound, OSError,\\nPdfReadError, or ValueError"]
    ok -->|yes| hid["document_id = SHA-256\\ndocument_name = file name"]
    hid --> lineage["Crawled files also keep\\nURL, parent URL, and depth"]
    lineage --> json["Write one artifact JSON\\nunder data/extracted"]
    json --> later["python run.py index\\nembeds chunks, summaries,\\nand composites"]
    later --> live["Live index shape:\\n2,880 PDF chunks\\n2,880 summaries\\n316 composites"]
    pipe["PDFRAGPipeline.ingest_pdf\\nembeds now and skips summaries"]
    proc -.-> pipe
`,
  extract: `stateDiagram-v2
    [*] --> Opened
    Opened --> PageFailed: exception on this page only
    Opened --> Cleaned: pypdf extract_text
    Cleaned --> TextPage: 15 or more non-space characters
    Cleaned --> Sparse: under 15 non-space characters
    Sparse --> Recovered: Tesseract returns usable text
    Sparse --> ImageOnly: no Tesseract, or OCR still sparse
    TextPage --> Blocks: heading or paragraph
    Recovered --> Blocks
    ImageOnly --> Blocks: page is kept and flagged
    PageFailed --> Next
    Blocks --> Next
    Next --> [*]
`,
  lineage: `erDiagram
    PDF ||--|{ PAGE : contains
    PAGE ||--|{ BLOCK : "splits into"
    BLOCK }o--|{ CHUNK : "packed into"
    CHUNK ||--o| SUMMARY : "CLI extractive summary"
    PDF {
      string document_id "SHA-256 of the bytes"
      string document_name "file name"
      string source_kind "crawled_pdf or uploaded_pdf"
    }
    CHUNK {
      string chunk_id "also the Elasticsearch id"
      string section "heading, else Introduction"
      string page_numbers "pages of the blocks in the group"
      string lineage "path, url, parent, depth"
    }
    SUMMARY {
      string summary_id "sum_ plus the chunk id"
      string page_numbers "not copied, so the list is empty"
    }
`,
  embed: `flowchart TD
    cfg{"embedding.provider"}
    cfg -->|local, active| local["Load all-MiniLM-L6-v2<br/>refuse to start if width is not 384"]
    cfg -->|mock| mock["Deterministic 384 random values<br/>for tests, not for meaning"]
    cfg -->|custom_api| api["POST to EMBEDDING_ENDPOINT"]
    local --> encode["encode one question<br/>or a batch of 32 chunks"]
    mock --> encode
    api --> encode
    encode --> store["Store the raw floats<br/>cosine is applied later by Elasticsearch"]
`,
  elastic: `classDiagram
    class IndexDocument {
      keyword chunk_id
      keyword document_id
      keyword source_kind
      text content
      integer page_numbers
      keyword figure_refs
      object lineage
      dense_vector embedding
      keyword schema_version
    }
    note for IndexDocument "384 dimensions, cosine, index true. schema_version is v3. The same chunk id replaces the document."
`,
  vector: `sequenceDiagram
    autonumber
    participant Q as Question
    participant V as VectorRetriever
    participant M as MiniLM
    participant ES as Elasticsearch HNSW
    Q->>V: search text
    V->>M: embed_text, same model as indexing
    M-->>V: 384 floats
    V->>ES: knn on field embedding, k and num_candidates
    Note over ES: cosine similarity<br/>num_candidates must be at least k
    ES-->>V: approximate neighbours
    V-->>Q: hits truncated to k
`,
  query: `sequenceDiagram
    autonumber
    participant Demo as _ask.py
    participant Ask as answer_question
    participant Hy as HybridRetriever
    participant ES as rag_corpus_v1
    participant Gate as grounding_failure
    participant LLM as Llama 3.2
    Demo->>Ask: What cyber forensics work does C-DAC describe?
    Ask->>Hy: mode hybrid, pool 20, candidates 50
    Hy->>ES: BM25 match and cosine kNN
    ES-->>Hy: two score lists
    Hy-->>Ask: 5 hits, including chunk_002 page 1 at 0.6678
    Ask->>Gate: leading score against 0.35
    Gate-->>Ask: continue, no role word in the question
    Ask->>Ask: append 5 knowledge-graph hits
    Ask->>LLM: system rules, unified context, temperature 0
    LLM-->>Demo: q5.json, abstained false, 10 sources
    Note over Demo: The PDF was already indexed.<br/>This call did not re-read the file.
`,
  tests: `mindmap
    root((Test run<br/>281 passed))
      Unit and contract
        Chunk split, overlap, merge
        Fusion weights must sum to 1
        Prompt text and abstain sentence
        Mock vectors and a stand-in client
      Teardown
        7 errors
        WinError 32
        temp PDF still open
      Not proved by the suite
        Tesseract was not on PATH
        31 sparse pages not re-read
        No labelled hit rate on real ids
        HNSW latency not saved
`,
};

function run(cmd, args) {
  const result = spawnSync(cmd, args, {
    encoding: "utf8",
    shell: cmd === "npx",
  });
  if (result.status !== 0) {
    throw new Error(`${cmd} ${args.join(" ")}\n${result.stdout}\n${result.stderr}`);
  }
}

function crop(png) {
  const code = `
from PIL import Image
p = r"""${png}"""
im = Image.open(p).convert("RGB")
px = im.load()
w, h = im.size
bg = (244, 241, 234)
minx, miny, maxx, maxy = w, h, 0, 0
for y in range(0, h, 2):
    for x in range(0, w, 2):
        r, g, b = px[x, y]
        if abs(r-bg[0]) + abs(g-bg[1]) + abs(b-bg[2]) > 28:
            if x < minx: minx = x
            if y < miny: miny = y
            if x > maxx: maxx = x
            if y > maxy: maxy = y
if maxx > minx:
    im.crop((max(0, minx-16), max(0, miny-16), min(w, maxx+18), min(h, maxy+18))).save(p)
`;
  const py = join(root, "..", "..", ".venv", "Scripts", "python.exe");
  run(py, ["-c", code]);
}

function shoot(htmlPath, pngPath, width, height) {
  const pad = 48;
  run(chrome, [
    "--headless=new",
    "--disable-gpu",
    "--hide-scrollbars",
    "--force-device-scale-factor=2",
    `--window-size=${Math.ceil(width + pad)},${Math.ceil(height + pad)}`,
    `--screenshot=${pngPath}`,
    pathToFileURL(htmlPath).href,
  ]);
  crop(pngPath);
}

const graphviz = await Graphviz.load();
for (const [key, dot] of Object.entries(graphs)) {
  const svg = graphviz.dot(dot);
  const svgPath = join(srcDir, `${key}.svg`);
  writeFileSync(svgPath, svg);
  const widthPt = Number((svg.match(/width="([\d.]+)/) || [])[1] || 800);
  const heightPt = Number((svg.match(/height="([\d.]+)/) || [])[1] || 600);
  const width = Math.ceil(widthPt * 96 / 72);
  const height = Math.ceil(heightPt * 96 / 72);
  const sized = svg
    .replace(/width="[\d.]+pt"/, `width="${width}"`)
    .replace(/height="[\d.]+pt"/, `height="${height}"`);
  const htmlPath = join(srcDir, `${key}.html`);
  writeFileSync(
    htmlPath,
    `<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;background:#f4f1ea}</style></head><body>${sized}</body></html>`,
  );
  const png = join(outDir, `${key}.png`);
  shoot(htmlPath, png, width, height);
  console.log("graphviz", key);
}

for (const [key, source] of Object.entries(mermaid)) {
  const mmd = join(srcDir, `${key}.mmd`);
  const png = join(outDir, `${key}.png`);
  writeFileSync(mmd, source.trim() + "\n");
  run("npx", [
    "-y",
    "@mermaid-js/mermaid-cli",
    "-i",
    mmd,
    "-o",
    png,
    "-c",
    themePath,
    "-t",
    "base",
    "-b",
    "#f4f1ea",
    "-s",
    "2",
  ]);
  crop(png);
  console.log("mermaid", key);
}

console.log("done", outDir);
