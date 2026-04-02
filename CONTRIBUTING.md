# Contributing to DocuMind AI

Thanks for taking the time to contribute. This project aims to stay clean, professional, and industry-aligned, so a few simple rules help keep things smooth.

## How to Contribute

- Open an issue for bugs or feature requests.
- Create a feature branch for every change.
- Keep PRs focused and small.
- Update docs when behavior or APIs change.

## Development Setup

Backend:
```bash
cd /Users/joyaljoy/source/ai-doc-search/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Frontend:
```bash
cd /Users/joyaljoy/source/ai-doc-search/frontend
npm install
npm run dev
```

## Branching & Commits

- Use `feature/`, `fix/`, or `chore/` prefixes.
- Write clear, imperative commit messages:
  - Good: "Add /query endpoint for semantic search"
  - Bad: "updates"

## Code Style

- Keep code readable and simple.
- Match the existing code conventions.
- Run `npm run lint` for frontend changes.
- Manually smoke-test API endpoints for backend changes.

## PR Checklist

- [ ] No secrets or `.env` files committed
- [ ] `README.md` updated if behavior or API changes
- [ ] UI changes include a short description or screenshot
- [ ] Code runs locally

## License

By contributing, you agree that your contributions are licensed under Apache-2.0.
