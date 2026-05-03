# Contributing to Nexuss-Transformer

Thank you for your interest in contributing to **Nexuss-Transformer**! This project aims to advance Amharic and Ge'ez language processing through state-of-the-art transformer models. We welcome contributions from the community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Reporting Issues](#reporting-issues)
- [Community](#community)

## Code of Conduct

Please be respectful and inclusive in all interactions. We are committed to providing a welcoming environment for contributors of all backgrounds.

## How to Contribute

### 1. Fork the Repository
Click the "Fork" button on the [GitHub repository](https://github.com/nexuss0781/Nexuss-Transformer) to create your own copy.

### 2. Clone Your Fork
```bash
git clone https://github.com/your-username/Nexuss-Transformer.git
cd Nexuss-Transformer
```

### 3. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 4. Make Your Changes
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 5. Commit Your Changes
```bash
git add .
git commit -m "feat: add your feature description"
```

We follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for code style changes
- `refactor:` for code refactoring
- `test:` for adding tests
- `chore:` for maintenance tasks

### 6. Push and Create a Pull Request
```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub with a clear description of your changes.

## Development Setup

### Prerequisites
- Python 3.8+
- Git
- pip or conda

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Running Tests
```bash
pytest tests/
```

## Pull Request Guidelines

- **One PR per feature/fix**: Keep pull requests focused on a single change
- **Clear title and description**: Explain what your PR does and why
- **Reference issues**: Link to any related issues using `#issue-number`
- **Pass CI checks**: Ensure all tests pass and linting is clean
- **Review feedback**: Be responsive to review comments and make requested changes

## Reporting Issues

When reporting issues, please include:
- A clear, descriptive title
- Steps to reproduce the problem
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
- Code snippets or error messages if applicable

## Community

Join our community discussions on:
- [GitHub Discussions](https://github.com/nexuss0781/Nexuss-Transformer/discussions)
- [Hugging Face Model Page](https://huggingface.co/Nexuss0781/Nexuss-Transformer)

## License

By contributing to this project, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

Thank you for helping make Nexuss-Transformer better for everyone! 🚀
