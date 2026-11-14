import { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

type Project = {
  id: string;
  number: string;
  title: string;
  shortTitle: string;
  category: 'Classification' | 'Regression' | 'NLP';
  eyebrow: string;
  description: string;
  outcome: string;
  outcomeLabel: string;
  dataset: string;
  methods: string[];
  accent: string;
  preview: string;
  previewAlt: string;
  icon: 'flower' | 'home' | 'pulse' | 'text';
  readme: string;
  notebook: string;
  source: string;
  folder: string;
  highlights: string[];
  caveat: string;
};

const REPO = 'https://github.com/Subiswas36218/Datheon_Machine_Learning_Data_Science_Internship';

const projects: Project[] = [
  {
    id: 'iris', number: '01', title: 'Iris Species Classification', shortTitle: 'Iris Classification',
    category: 'Classification', eyebrow: 'SUPERVISED LEARNING · MULTICLASS',
    description: 'A compact classification study that turns four flower measurements into species predictions, with interpretable model comparisons and diagnostic plots.',
    outcome: '93.33%', outcomeLabel: 'test accuracy · both models',
    dataset: '150 observations · 4 measurements · 3 species',
    methods: ['Python', 'Pandas', 'scikit-learn', 'Logistic Regression', 'Decision Tree', 'PCA'],
    accent: 'lime', preview: '/previews/iris-pairplot.png', previewAlt: 'Iris feature pairplot grouped by species', icon: 'flower',
    folder: 'iris-classification-pipeline',
    readme: `${REPO}/blob/main/iris-classification-pipeline/README.md`,
    notebook: `${REPO}/blob/main/iris-classification-pipeline/notebooks/iris_classification.ipynb`,
    source: `${REPO}/blob/main/iris-classification-pipeline/src/iris_classification.py`,
    highlights: ['Stratified train/test evaluation with a fixed random seed.', 'Standardized Logistic Regression and Decision Tree pipelines.', 'PCA, multiclass ROC-AUC, precision–recall, learning curves and confusion matrices.'],
    caveat: 'The reported score comes from one 30-row held-out test partition; it is a reproducible classroom benchmark, not a confidence interval.'
  },
  {
    id: 'housing', number: '02', title: 'California Housing Price Regression', shortTitle: 'Housing Price Regression',
    category: 'Regression', eyebrow: 'REGRESSION · REGULARIZATION',
    description: 'A tabular regression workflow comparing ordinary least squares with L1 and L2 regularization, including preprocessing, feature engineering and alpha search.',
    outcome: '$67,010', outcomeLabel: 'OLS test RMSE · USD',
    dataset: '20,640 records · 10 source columns · 1 target',
    methods: ['Python', 'Pandas', 'scikit-learn', 'Linear Regression', 'Ridge', 'Lasso', 'GridSearchCV'],
    accent: 'blue', preview: '/previews/housing-predicted-vs-actual.png', previewAlt: 'Predicted versus actual California house values', icon: 'home',
    folder: 'california-housing-ridge-lasso',
    readme: `${REPO}/blob/main/california-housing-ridge-lasso/README.md`,
    notebook: `${REPO}/blob/main/california-housing-ridge-lasso/notebooks/real_estate_housing_price_prediction.ipynb`,
    source: `${REPO}/blob/main/california-housing-ridge-lasso/src/housing_price_pipeline.py`,
    highlights: ['Numeric and categorical preprocessing in a reusable pipeline.', 'Missing-value imputation and one-hot encoding for ocean proximity.', 'Cross-validated Ridge/Lasso alpha selection and residual diagnostics.'],
    caveat: 'OLS was marginally best on this fixed split (R² 0.6573); Ridge was nearly tied. The target ceiling and residual patterns limit interpretation at the high end.'
  },
  {
    id: 'churn', number: '03', title: 'Telco Customer Churn Classifier', shortTitle: 'Customer Churn',
    category: 'Classification', eyebrow: 'BINARY CLASSIFICATION · CUSTOMER ANALYTICS',
    description: 'A churn-risk modelling workflow that combines categorical and numerical customer signals, leakage-safe resampling, model comparison and threshold analysis.',
    outcome: '0.8355', outcomeLabel: 'XGBoost ROC-AUC',
    dataset: '7,043 customers · 21 columns · binary churn target',
    methods: ['Python', 'Pandas', 'scikit-learn', 'SMOTENC', 'Random Forest', 'XGBoost'],
    accent: 'coral', preview: '/previews/churn-roc-curves.png', previewAlt: 'ROC curve comparison for churn classifiers', icon: 'pulse',
    folder: 'telco-customer-churn-classifier',
    readme: `${REPO}/blob/main/telco-customer-churn-classifier/README.md`,
    notebook: `${REPO}/blob/main/telco-customer-churn-classifier/notebooks/customer_churn_classification.ipynb`,
    source: `${REPO}/blob/main/telco-customer-churn-classifier/src/churn_pipeline.py`,
    highlights: ['Customer churn exploration across tenure, contract, payment and service features.', 'SMOTENC applied inside the training/CV pipeline to protect validation and test data.', 'Random Forest vs XGBoost with ROC, precision–recall and threshold-sensitivity analysis.'],
    caveat: 'XGBoost led on ROC-AUC, while Random Forest was slightly stronger on several default-threshold metrics. Threshold choice depends on retention costs and capacity.'
  },
  {
    id: 'sentiment', number: '04', title: 'Customer Review Sentiment Classification', shortTitle: 'Review Sentiment · NLP',
    category: 'NLP', eyebrow: 'NATURAL LANGUAGE PROCESSING · TF-IDF',
    description: 'A small-sample text classification study using NLTK normalization and TF-IDF features to compare Multinomial Naive Bayes with Logistic Regression.',
    outcome: '44.44%', outcomeLabel: 'out-of-fold accuracy · 9 binary reviews',
    dataset: '10 reviews total · 5 positive · 4 negative · 1 neutral',
    methods: ['Python', 'NLTK', 'TF-IDF', 'scikit-learn', 'Naive Bayes', 'Logistic Regression'],
    accent: 'violet', preview: '/previews/sentiment-confusion-matrices.png', previewAlt: 'Out-of-fold confusion matrices for review sentiment models', icon: 'text',
    folder: 'customer-review-sentiment-classifier-project',
    readme: `${REPO}/blob/main/customer-review-sentiment-classifier-project/README.md`,
    notebook: `${REPO}/blob/main/customer-review-sentiment-classifier-project/notebooks/customer_review_sentiment_classification.ipynb`,
    source: `${REPO}/blob/main/customer-review-sentiment-classifier-project/src/sentiment_pipeline.py`,
    highlights: ['Text cleaning, tokenization, stopword handling and lemmatization support.', 'TF-IDF vectorization with unigrams and bigrams.', 'Stratified 3-fold out-of-fold comparison and custom prediction helper.'],
    caveat: 'Only nine reviews have binary labels; the single neutral example is not enough to train a neutral class. Scores are a pipeline demonstration, not reliable production evidence.'
  }
];

const filters = ['All projects', 'Classification', 'Regression', 'NLP'] as const;
type Filter = typeof filters[number];

function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const common = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.7, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const, 'aria-hidden': true as const };
  switch (name) {
    case 'arrow': return <svg {...common}><path d="M5 12h14M13 5l7 7-7 7" /></svg>;
    case 'external': return <svg {...common}><path d="M14 4h6v6M20 4l-9 9"/><path d="M18 13v6a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h6"/></svg>;
    case 'github': return <svg {...common}><path d="M9 19c-4.3 1.4-4.3-2.5-6-3m12 6v-3.9a3.4 3.4 0 0 0-.9-2.7c3-.3 6.1-1.5 6.1-6.7a5.2 5.2 0 0 0-1.4-3.6 4.8 4.8 0 0 0-.1-3.6S17.5 1 15 3.2a13.3 13.3 0 0 0-7 0C5.5 1 3.3 1.5 3.3 1.5a4.8 4.8 0 0 0-.1 3.6A5.2 5.2 0 0 0 1.8 8.7c0 5.2 3.1 6.4 6.1 6.7A3.4 3.4 0 0 0 7 18.1V22"/></svg>;
    case 'code': return <svg {...common}><path d="m8 17-5-5 5-5M16 7l5 5-5 5M14 4l-4 16"/></svg>;
    case 'book': return <svg {...common}><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v17H6.5A2.5 2.5 0 0 1 4 17.5z"/><path d="M4 17.5A2.5 2.5 0 0 1 6.5 15H20M8 7h8M8 10h6"/></svg>;
    case 'flower': return <svg {...common}><path d="M12 12c-3-5-7-5-7-1 0 2 2 3 7 3M12 12c3-5 7-5 7-1 0 2-2 3-7 3M12 12c-5 3-5 7-1 7 2 0 3-2 3-7M12 12c5 3 5 7 1 7-2 0-3-2-3-7"/><circle cx="12" cy="12" r="2"/></svg>;
    case 'home': return <svg {...common}><path d="m3 10 9-7 9 7v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z"/><path d="M9 21v-7h6v7M3 10h18"/></svg>;
    case 'pulse': return <svg {...common}><path d="M3 12h4l3-8 4 16 3-8h4"/></svg>;
    case 'text': return <svg {...common}><path d="M4 6V4h16v2M12 4v16M8 20h8M7 9h10"/></svg>;
    case 'spark': return <svg {...common}><path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="m19 16 .8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8z"/></svg>;
    case 'close': return <svg {...common}><path d="m18 6-12 12M6 6l12 12"/></svg>;
    case 'layers': return <svg {...common}><path d="m12 3 9 5-9 5-9-5zM3 12l9 5 9-5M3 16l9 5 9-5"/></svg>;
    default: return <svg {...common}><circle cx="12" cy="12" r="8"/></svg>;
  }
}

function ProjectGlyph({ project }: { project: Project }) {
  return <span className={`project-glyph ${project.accent}`}><Icon name={project.icon} size={26} /></span>;
}

function App() {
  const [filter, setFilter] = useState<Filter>('All projects');
  const [selected, setSelected] = useState<Project | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const visibleProjects = useMemo(() => filter === 'All projects' ? projects : projects.filter((p) => p.category === filter), [filter]);

  useEffect(() => {
    if (!selected) return;
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === 'Escape') setSelected(null); };
    window.addEventListener('keydown', onKeyDown);
    document.body.style.overflow = 'hidden';
    return () => { window.removeEventListener('keydown', onKeyDown); document.body.style.overflow = ''; };
  }, [selected]);

  const scrollToProjects = () => document.getElementById('projects')?.scrollIntoView({ behavior: 'smooth' });

  return (
    <div className="site-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Subhankar Biswas home"><span className="brand-mark">SB<span>.</span></span><span className="brand-caption">DATA / ML PORTFOLIO</span></a>
        <button className="menu-toggle" onClick={() => setMenuOpen((v) => !v)} aria-label="Toggle navigation" aria-expanded={menuOpen}><span/><span/></button>
        <nav className={menuOpen ? 'main-nav open' : 'main-nav'}>
          <a href="#projects" onClick={() => setMenuOpen(false)}>Projects <span>04</span></a>
          <a href="#approach" onClick={() => setMenuOpen(false)}>Approach</a>
          <a href="#about" onClick={() => setMenuOpen(false)}>About</a>
          <a className="nav-github" href={REPO} target="_blank" rel="noreferrer" onClick={() => setMenuOpen(false)}>GitHub <Icon name="external" size={14}/></a>
        </nav>
      </header>

      <main id="top">
        <section className="hero section-wrap">
          <div className="hero-left">
            <div className="overline"><span className="status-dot"/> PROJECT COLLECTION <span className="overline-rule"/> 2026</div>
            <h1>Data into<br/><em>decisions.</em></h1>
            <p className="hero-copy">I’m <strong>Subhankar Biswas</strong> — exploring the space where data engineering meets applied machine learning. Four focused projects, built from raw datasets to tested models and interpretable results.</p>
            <div className="hero-actions">
              <button className="button button-primary" onClick={scrollToProjects}>Explore the work <Icon name="arrow" size={17}/></button>
              <a className="button button-ghost" href={REPO} target="_blank" rel="noreferrer"><Icon name="github" size={17}/> View repository</a>
            </div>
            <div className="hero-note"><span className="note-line"/> <span>FOUR PROJECTS. ONE REPOSITORY. END-TO-END WORKFLOWS.</span></div>
          </div>
          <div className="hero-visual" aria-label="Portfolio project overview graphic">
            <div className="visual-topline"><span>FIELD NOTES / 001—004</span><span>42° 21′ N · DATA MODE</span></div>
            <div className="orbit orbit-outer"/><div className="orbit orbit-inner"/>
            <div className="visual-center"><div className="center-cross">+</div><div className="center-number">04</div><div className="center-label">PROJECTS<br/>IN FOCUS</div></div>
            <div className="orbit-node node-one"><span className="node-number">01</span><span className="node-name">CLASSIFY</span></div>
            <div className="orbit-node node-two"><span className="node-number">02</span><span className="node-name">PREDICT</span></div>
            <div className="orbit-node node-three"><span className="node-number">03</span><span className="node-name">DETECT</span></div>
            <div className="orbit-node node-four"><span className="node-number">04</span><span className="node-name">UNDERSTAND</span></div>
            <div className="visual-footer"><span>DATA → FEATURES → MODELS</span><span className="visual-coord">SB / 2026</span></div>
          </div>
          <div className="hero-index"><span>SCROLL TO EXPLORE</span><span className="index-line"/><span>01 / 04</span></div>
        </section>

        <section className="metrics-strip" aria-label="Portfolio summary">
          <div className="metrics-inner">
            <div className="metric-intro">A SMALL, FOCUSED<br/>BODY OF WORK</div>
            <div className="metric"><strong>04</strong><span>PROJECTS</span></div>
            <div className="metric"><strong>04</strong><span>DATASETS</span></div>
            <div className="metric"><strong>03</strong><span>ML DOMAINS</span></div>
            <div className="metric metric-last"><strong>01</strong><span>REPOSITORY</span></div>
          </div>
        </section>

        <section className="projects-section section-wrap" id="projects">
          <div className="section-heading">
            <div><div className="section-kicker">SELECTED WORK <span>— 2026</span></div><h2>Four tasks.<br/><em>Four ways to think.</em></h2></div>
            <p className="section-intro">A compact portfolio spanning supervised classification, regularized regression, customer analytics and natural language processing. Each card links directly to its project documentation, notebook and source code.</p>
          </div>
          <div className="filter-row" role="group" aria-label="Filter projects by discipline">
            <span className="filter-label">FILTER BY</span>
            {filters.map((item) => <button key={item} className={filter === item ? 'filter-chip active' : 'filter-chip'} onClick={() => setFilter(item)} aria-pressed={filter === item}>{item}<span>{item === 'All projects' ? '04' : String(projects.filter((p) => p.category === item).length).padStart(2, '0')}</span></button>)}
            <span className="filter-count">SHOWING {String(visibleProjects.length).padStart(2, '0')} / 04</span>
          </div>
          <div className="project-grid">
            {visibleProjects.map((project) => <article className={`project-card ${project.accent}`} key={project.id}>
              <div className="card-top"><span className="card-number">TASK / {project.number}</span><span className="card-category">{project.category}</span></div>
              <div className="card-visual-row"><ProjectGlyph project={project}/><span className="card-eyebrow">{project.eyebrow}</span><span className="card-orbit">{project.number}</span></div>
              <div className="card-preview"><img src={project.preview} alt={project.previewAlt} loading="lazy"/><span>MODEL OUTPUT / FIGURE</span></div>
              <h3>{project.title}</h3>
              <p className="card-description">{project.description}</p>
              <div className="card-outcome"><div><strong>{project.outcome}</strong><span>{project.outcomeLabel}</span></div><div className="outcome-arrow"><Icon name="arrow" size={18}/></div></div>
              <div className="dataset-line"><span className="tiny-square"/>{project.dataset}</div>
              <div className="tag-list">{project.methods.slice(0, 5).map((method) => <span key={method}>{method}</span>)}{project.methods.length > 5 && <span className="tag-more">+{project.methods.length - 5}</span>}</div>
              <div className="card-footer"><button className="text-link" onClick={() => setSelected(project)}>Project details <Icon name="arrow" size={16}/></button><a href={project.readme} target="_blank" rel="noreferrer" className="card-repo-link">OPEN TASK <Icon name="external" size={13}/></a></div>
            </article>)}
          </div>
          <div className="projects-footnote"><span className="footnote-star">✳</span><span>All project notebooks, supplied datasets, figures, reports and reproducible source files live in the same public repository.</span><a href={REPO} target="_blank" rel="noreferrer">BROWSE ALL FILES <Icon name="arrow" size={14}/></a></div>
        </section>

        <section className="approach-section" id="approach">
          <div className="section-wrap approach-inner">
            <div className="approach-title"><div className="section-kicker">HOW I WORK <span>— THE PROCESS</span></div><h2>Curiosity, with<br/><em>a clear method.</em></h2><p>Across different problem types, the process stays grounded: understand the data, build a sound baseline, test choices and communicate what the results do — and do not — say.</p></div>
            <div className="process-list">
              <div className="process-item"><span className="process-index">01</span><div><h3>Inspect the signal</h3><p>Audit the dataset, validate fields, explore distributions and identify data-quality constraints before modelling.</p></div><span className="process-code">DATA / EDA</span></div>
              <div className="process-item"><span className="process-index">02</span><div><h3>Build a fair pipeline</h3><p>Use appropriate preprocessing, train/test separation and fold-aware transformations to reduce leakage.</p></div><span className="process-code">PIPELINE / CV</span></div>
              <div className="process-item"><span className="process-index">03</span><div><h3>Compare, don't assume</h3><p>Evaluate more than one model and use metrics that fit the task — not just a single headline score.</p></div><span className="process-code">MODEL / METRICS</span></div>
              <div className="process-item"><span className="process-index">04</span><div><h3>Explain the outcome</h3><p>Use plots, reports and limitations to make model behaviour understandable and reproducible.</p></div><span className="process-code">INSIGHT / REVIEW</span></div>
            </div>
          </div>
        </section>

        <section className="about-section section-wrap" id="about">
          <div className="about-top"><div className="section-kicker">ABOUT THIS COLLECTION <span>— 04 / 04</span></div><span className="about-stamp">BUILT WITH<br/>OPEN TOOLS</span></div>
          <div className="about-content"><h2>Applied learning.<br/><em>Documented end to end.</em></h2><div className="about-copy"><p>This portfolio brings together four machine-learning internship tasks in one place. It is designed to make the work easy to review: each project has its own dataset, notebook, source, outputs, tests and workflow.</p><p>The goal is not to overstate a metric. It is to show the decisions behind it — including where a small sample or a fixed test split limits what can be concluded.</p><a className="about-link" href={REPO} target="_blank" rel="noreferrer">Explore the GitHub repository <Icon name="external" size={15}/></a></div></div>
          <div className="toolbelt"><span className="toolbelt-label">TOOLS ACROSS THE FOUR PROJECTS</span><div className="toolbelt-items"><span>PYTHON</span><span>PANDAS</span><span>NUMPY</span><span>SCIKIT-LEARN</span><span>JUPYTER</span><span>MATPLOTLIB</span><span>SEABORN</span><span>XGBOOST</span><span>NLTK</span><span>GITHUB ACTIONS</span></div></div>
        </section>
      </main>

      <footer className="site-footer section-wrap">
        <div className="footer-top"><div className="footer-brand"><span className="brand-mark">SB<span>.</span></span><p>DATA / MACHINE LEARNING<br/>PORTFOLIO — 2026</p></div><a href="#top" className="back-top">BACK TO TOP <span>↑</span></a></div>
        <div className="footer-bottom"><span>© 2026 SUBHANKAR BISWAS</span><span>FOCUSED ON FOUR INTERNSHIP PROJECTS</span><a href={REPO} target="_blank" rel="noreferrer"><Icon name="github" size={14}/> GITHUB REPOSITORY</a></div>
      </footer>

      {selected && <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setSelected(null); }}>
        <section className="project-modal" role="dialog" aria-modal="true" aria-labelledby="modal-title">
          <div className="modal-top"><span>TASK / {selected.number} — PROJECT BRIEF</span><button className="modal-close" onClick={() => setSelected(null)} aria-label="Close details"><Icon name="close" size={19}/></button></div>
          <div className="modal-heading"><ProjectGlyph project={selected}/><div><span className="section-kicker">{selected.eyebrow}</span><h2 id="modal-title">{selected.title}</h2></div></div>
          <p className="modal-description">{selected.description}</p>
          <div className="modal-preview"><img src={selected.preview} alt={selected.previewAlt}/></div>
          <div className="modal-stat"><strong>{selected.outcome}</strong><span>{selected.outcomeLabel}</span><small>{selected.dataset}</small></div>
          <div className="modal-columns"><div><h3>WHAT'S INSIDE</h3><ul>{selected.highlights.map((item) => <li key={item}>{item}</li>)}</ul></div><div><h3>INTERPRETATION NOTE</h3><p className="modal-caveat">{selected.caveat}</p><h3>TECH STACK</h3><div className="tag-list modal-tags">{selected.methods.map((method) => <span key={method}>{method}</span>)}</div></div></div>
          <div className="modal-links"><a href={selected.readme} target="_blank" rel="noreferrer"><Icon name="book" size={16}/> README <Icon name="external" size={13}/></a><a href={selected.notebook} target="_blank" rel="noreferrer"><Icon name="code" size={16}/> NOTEBOOK <Icon name="external" size={13}/></a><a href={selected.source} target="_blank" rel="noreferrer"><Icon name="layers" size={16}/> SOURCE <Icon name="external" size={13}/></a></div>
        </section>
      </div>}
    </div>
  );
}

export default App;

// Mount the React application into the root element declared in index.html.
// Without this call Vite can serve a valid page shell, but the browser remains blank.
const rootElement = document.getElementById('root');

if (!rootElement) {
  throw new Error('Root element #root was not found. Check index.html.');
}

createRoot(rootElement).render(<App />);

