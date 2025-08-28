# Contributing to workflow_16s

Thank you for your interest in contributing to workflow_16s! This document provides guidelines for contributing to this 16S rRNA gene analysis pipeline.

## Development Setup

### Prerequisites
- Python 3.10+
- Conda or Mamba package manager
- Git

### Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/heathermacgregor/workflow_16s.git
   cd workflow_16s
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f workflow_16s.yml
   conda activate workflow_16s
   ```

3. Install in development mode:
   ```bash
   pip install -e .
   ```

## Contributing Process

### 1. Issue Reporting
- Search existing issues before creating new ones
- Use clear, descriptive titles
- Include:
  - Environment details (OS, Python version, conda environment)
  - Steps to reproduce
  - Expected vs actual behavior
  - Relevant log outputs

### 2. Feature Requests
- Describe the proposed feature clearly
- Explain the biological/analytical motivation
- Consider backward compatibility
- Provide example use cases

### 3. Pull Requests
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes following our coding standards
4. Add or update tests as needed
5. Update documentation
6. Commit with clear, descriptive messages
7. Push to your fork and submit a pull request

## Coding Standards

### Python Style
- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Add docstrings for all public functions and classes
- Use meaningful variable and function names

### Code Organization
- Keep functions focused and single-purpose
- Use appropriate error handling and logging
- Follow the existing project structure
- Add comments for complex logic

### Testing
- Write unit tests for new functionality
- Ensure all tests pass before submitting
- Aim for good test coverage
- Include integration tests for workflow components

## Documentation Guidelines

### Code Documentation
- Use Google-style docstrings
- Document all parameters, return values, and exceptions
- Provide usage examples for complex functions

### User Documentation
- Update README.md for user-facing changes
- Add examples to docs/ directory
- Update configuration documentation as needed

## Biological Context

When contributing to this 16S rRNA analysis pipeline, consider:
- **Scientific accuracy**: Ensure methods are biologically sound
- **Reproducibility**: All analyses should be reproducible
- **Standard practices**: Follow established bioinformatics conventions
- **Data formats**: Support standard formats (BIOM, QIIME 2, etc.)

## Review Process

### What We Look For
- **Functionality**: Does the code work as intended?
- **Testing**: Are there adequate tests?
- **Documentation**: Is the code well-documented?
- **Performance**: Does it maintain reasonable performance?
- **Compatibility**: Does it work across supported environments?

### Response Time
- We aim to respond to issues within 1 week
- Pull requests typically reviewed within 2 weeks
- Complex changes may require additional review time

## Getting Help

- **Documentation**: Check the docs/ directory
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for sensitive issues

## Recognition

Contributors will be acknowledged in:
- GitHub contributors list
- CHANGELOG.md for significant contributions
- README.md for major feature additions

## Code of Conduct

This project follows a Code of Conduct. By participating, you agree to uphold professional and respectful behavior in all interactions.

---

Thank you for contributing to workflow_16s! Your contributions help make 16S rRNA analysis more accessible and reproducible for the research community.