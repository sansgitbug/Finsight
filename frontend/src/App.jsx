import { useEffect, useState } from "react";
import "./App.css";
import ReactMarkdown from "react-markdown";

const API = "http://127.0.0.1:8000";

function ScoreBar({ value, max = 5 }) {
  const width = Math.min((Math.abs(value) / max) * 100, 100);

  return (
    <div className="score-bar">
      <div
        className="score-bar-fill"
        style={{ width: `${width}%` }}
      />
    </div>
  );
}

function ShapBlock({ shap }) {
  if (!shap || shap.error) {
    return (
      <div className="shap-block">
        <div className="shap-title">RELEVANCE ATTRIBUTION</div>
        <div className="shap-unavailable">
          SHAP unavailable
        </div>
      </div>
    );
  }

  return (
    <div className="shap-block">
      <div className="shap-title">
        RELEVANCE ATTRIBUTION
      </div>

      {shap.positive?.slice(0, 5).map((item, index) => (
        <div
          className="shap-row"
          key={`p-${item.token}-${index}`}
        >
          <span className="shap-token">
            {item.token}
          </span>

          <ScoreBar value={item.value} />

          <span className="shap-value positive">
            +{item.value.toFixed(2)}
          </span>
        </div>
      ))}

      {shap.negative?.slice(0, 3).map((item, index) => (
        <div
          className="shap-row"
          key={`n-${item.token}-${index}`}
        >
          <span className="shap-token">
            {item.token}
          </span>

          <ScoreBar value={item.value} />

          <span className="shap-value negative">
            {item.value.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  );
}

function SourceCard({ source }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="source-card">
      <button
        className="source-header"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="source-main">
          <div className="source-id">
            {source.chunk_id}
          </div>

          <div className="source-section">
            {source.section_name}
          </div>
        </div>

        <div className="source-score">
          <span>RERANK</span>
          {Number(source.reranker_score).toFixed(4)}
        </div>
      </button>

      <div className="source-meta">
        <span>{source.filing_date}</span>
        <span>{source.filing_type}</span>
        <span>
          DENSE #{source.dense_rank ?? "—"}
        </span>
        <span>
          BM25 #{source.bm25_rank ?? "—"}
        </span>
        <span>
          RRF {Number(source.rrf_score).toFixed(4)}
        </span>
      </div>

      <ShapBlock shap={source.shap} />

      {expanded && (
        <div className="source-text">
          {source.text}
        </div>
      )}
    </div>
  );
}

function Benchmark() {
  return (
    <footer className="benchmark">
      <div className="benchmark-title">
        RETRIEVAL BENCHMARK
      </div>

      <div className="benchmark-results">
        <div className="benchmark-item">
          <span>DENSE</span>
          <strong>30.6%</strong>
        </div>

        <span className="arrow">→</span>

        <div className="benchmark-item">
          <span>HYBRID</span>
          <strong>38.9%</strong>
        </div>

        <span className="arrow">→</span>

        <div className="benchmark-item highlight">
          <span>HYBRID + RERANKER</span>
          <strong>55.6%</strong>
        </div>
      </div>
    </footer>
  );
}

export default function App() {
  const [ticker, setTicker] = useState("");
  const [companies, setCompanies] = useState([]);

  const [query, setQuery] = useState("");

  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);

  const [loading, setLoading] = useState(false);
  const [ingesting, setIngesting] = useState(false);

  const [error, setError] = useState("");

  /*
   * Load companies that have already been ingested.
   * This does NOT restrict what ticker the user can enter.
   */
  useEffect(() => {
    loadCompanies();
  }, []);

  async function loadCompanies() {
    try {
      const response = await fetch(
        `${API}/api/companies`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
            "Unable to load companies"
        );
      }

      setCompanies(data.companies || []);
    } catch (err) {
      setError(err.message);
    }
  }

  /*
   * Ingest ANY ticker entered by the user.
   */
  async function ingestCompany() {
    const symbol = ticker.trim().toUpperCase();

    if (!symbol || ingesting) {
      return;
    }

    setIngesting(true);
    setError("");
    setAnswer("");
    setSources([]);

    try {
      const response = await fetch(
        `${API}/api/ingest`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            ticker: symbol,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
            `Unable to ingest ${symbol}`
        );
      }

      await loadCompanies();

    } catch (err) {
      setError(err.message);
    } finally {
      setIngesting(false);
    }
  }

  /*
   * Query the currently entered company.
   */
  async function runQuery() {
    const symbol = ticker.trim().toUpperCase();

    if (
      !symbol ||
      !query.trim() ||
      loading
    ) {
      return;
    }

    setLoading(true);
    setError("");
    setAnswer("");
    setSources([]);

    try {
      const response = await fetch(
        `${API}/api/query`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            ticker: symbol,
            query: query.trim(),
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
            "Query failed"
        );
      }

      setAnswer(data.answer || "");
      setSources(data.sources || []);

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  /*
   * Existing company metadata, if available.
   */
  const currentCompany = companies.find(
    (company) =>
      company.ticker ===
      ticker.trim().toUpperCase()
  );

  return (
    <div className="app">

      {/* =====================================================
          HEADER
          ===================================================== */}

      <header className="top-header">

        <div className="brand">
          FINSIGHT
        </div>

        <div className="company">
          <span className="ticker">
            {ticker
              ? ticker.toUpperCase()
              : "—"}
          </span>

          <span className="company-name">
            SEC RESEARCH
          </span>
        </div>

        {/* =================================================
            COMPANY INPUT
            ================================================= */}

        <div className="company-selector">

          <input
            className="ticker-input"
            value={ticker}
            onChange={(event) => {
              setTicker(
                event.target.value
                  .toUpperCase()
                  .replace(/[^A-Z0-9.-]/g, "")
              );

              setAnswer("");
              setSources([]);
              setError("");
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                ingestCompany();
              }
            }}
            placeholder="ENTER TICKER"
            maxLength={10}
            spellCheck={false}
          />

          <button
            className="ingest-button"
            onClick={ingestCompany}
            disabled={
              ingesting ||
              !ticker.trim()
            }
          >
            {ingesting
              ? "INGESTING..."
              : "INGEST COMPANY"}
          </button>

        </div>

        <div className="filing-info">

          <span>
            SEC FILINGS
          </span>

          <strong>
            {currentCompany
              ? `${currentCompany.filings} FILINGS`
              : "NOT INDEXED"}
          </strong>

        </div>

        <div className="status">

          <span className="status-dot" />

          LOCAL / OLLAMA

        </div>

      </header>

      {/* =====================================================
          WORKSPACE
          ===================================================== */}

      <main className="workspace">

        {/* =================================================
            LEFT PANEL
            ================================================= */}

        <section className="left-panel">

          <div className="panel-heading">
            <span>
              RESEARCH QUERY
            </span>

            <span className="panel-index">
              01
            </span>
          </div>

          <textarea
            value={query}
            onChange={(event) =>
              setQuery(event.target.value)
            }
            placeholder={
              ticker
                ? `Ask a question about ${ticker}...`
                : "Enter a ticker first..."
            }
          />

          <button
            className="run-button"
            onClick={runQuery}
            disabled={
              loading ||
              !ticker.trim() ||
              !query.trim()
            }
          >
            {loading
              ? "RUNNING..."
              : "RUN QUERY"}
          </button>

          <div className="evidence-heading">

            <span>
              EVIDENCE
            </span>

            <span>
              {sources.length} SOURCES
            </span>

          </div>

          {error && (
            <div className="error">
              {error}
            </div>
          )}

          <div className="sources">

            {sources.length === 0 &&
              !loading && (
                <div className="empty-evidence">
                  No evidence retrieved yet.
                </div>
              )}

            {sources.map((source) => (
              <SourceCard
                key={source.chunk_id}
                source={source}
              />
            ))}

          </div>

        </section>

        {/* =================================================
            RIGHT PANEL
            ================================================= */}

        <section className="right-panel">

          <div className="panel-heading">

            <span>
              ANALYSIS
            </span>

            <span className="panel-index">
              02
            </span>

          </div>

          <div className="answer">

            {answer ? (
              <div className="answer-text">
                <ReactMarkdown>
                  {answer}
                </ReactMarkdown>
              </div>
            ) : (
              <div className="empty-state">

                {ticker
                  ? `Submit a research query to generate grounded financial analysis for ${ticker}.`
                  : "Enter a company ticker and research question to begin."}

              </div>
            )}

          </div>

          {/* =================================================
              RETRIEVAL METRICS
              ================================================= */}

          <div className="metrics">

            <div className="panel-heading">
              <span>
                RETRIEVAL PERFORMANCE
              </span>
            </div>

            <div className="metric-grid">

              <div>
                <span>
                  METHOD
                </span>

                <strong>
                  HYBRID + RERANKER
                </strong>
              </div>

              <div>
                <span>
                  RECALL@5
                </span>

                <strong>
                  55.6%
                </strong>
              </div>

              <div>
                <span>
                  MRR
                </span>

                <strong>
                  0.375
                </strong>
              </div>

            </div>

            <div className="comparison">

              <div className="comparison-row">

                <span>
                  DENSE
                </span>

                <div className="comparison-track">
                  <div
                    style={{
                      width: "55%",
                    }}
                  />
                </div>

                <strong>
                  30.6%
                </strong>

              </div>

              <div className="comparison-row">

                <span>
                  BM25
                </span>

                <div className="comparison-track">
                  <div
                    style={{
                      width: "10%",
                    }}
                  />
                </div>

                <strong>
                  5.6%
                </strong>

              </div>

              <div className="comparison-row">

                <span>
                  HYBRID
                </span>

                <div className="comparison-track">
                  <div
                    style={{
                      width: "70%",
                    }}
                  />
                </div>

                <strong>
                  38.9%
                </strong>

              </div>

              <div className="comparison-row">

                <span>
                  RERANK
                </span>

                <div className="comparison-track">

                  <div
                    className="best"
                    style={{
                      width: "100%",
                    }}
                  />

                </div>

                <strong>
                  55.6%
                </strong>

              </div>

            </div>

          </div>

        </section>

      </main>

      <Benchmark />

    </div>
  );
}