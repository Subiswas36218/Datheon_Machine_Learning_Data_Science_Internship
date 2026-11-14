# Subhankar Biswas — Data & Machine Learning Portfolio

A responsive React + Vite portfolio featuring **only the four internship projects** in the parent monorepo.

## Included projects

1. Iris Species Classification — Logistic Regression and Decision Tree.
2. California Housing Price Regression — OLS, Ridge and Lasso.
3. Telco Customer Churn — Random Forest, XGBoost and SMOTENC.
4. Customer Review Sentiment — NLTK, TF-IDF, Multinomial Naive Bayes and Logistic Regression.

Every project card links directly to its README, primary notebook and reusable source code in the GitHub monorepo.

## Run locally

```bash
npm install
npm run dev
```

Production build:

```bash
npm run typecheck
npm run build
npm run preview
```

## Deploy on Vercel from the monorepo

Import `Subiswas36218/Datheon_Machine_Learning_Data_Science_Internship` into Vercel and configure:

| Setting | Value |
|---|---|
| Root Directory | `portfolio-site` |
| Framework Preset | Vite |
| Install Command | `npm install` |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Production Branch | `main` |

The `vercel.json` file contains the Vite build and output settings. The monorepo Root Directory must be set to `portfolio-site`, because Vercel builds the selected app directory in isolation.

See [Vercel's Vite guide](https://vercel.com/docs/frameworks/frontend/vite) and [Root Directory settings](https://vercel.com/docs/builds/configure-a-build#root-directory).
