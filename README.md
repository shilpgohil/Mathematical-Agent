# Mathematical Agent

## Introduction
Mathematical Agent is an advanced AI powered mathematics assistant that helps users solve complex mathematical problems across various domains. Built with Streamlit and powered by Google's Gemini model, it provides step by step solutions, detailed explanations, and access to a comprehensive mathematical knowledge base.

## Features
- **Multi-Domain Support**: Arithmetic, Algebra, Geometry, Calculus, Advanced Mathematics
- **Interactive Problem Solving**: Step by step solutions with detailed explanations
- **Knowledge Base Integration**: Built-in mathematical knowledge repository
- **Web Search Integration**: Ability to search for additional information when needed
- **LaTeX Support**: Clean and format LaTeX expressions

## Installation
1. Clone this repository:
```bash
git clone https://github.com/yourusername/mathematical-agent.git
```
2. Navigate to the project directory:
```bash
cd mathematical-agent
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Start the application:
```bash
streamlit run streamlit/app.py
```
2. Open your web browser and navigate to `http://localhost:8501`
3. Enter your mathematical problem in the input field
4. View the step-by-step solution and explanations

## Project Structure
```
mathematical-agent/
├── streamlit/
│   ├── app.py             # Main application file
│   ├── math_db/           # ChromaDB knowledge base
├── requirements.txt       # Python dependencies
├── Demo.mp4               # Demo Video 
├── README.md              # This documentation file
```

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments
- Google Gemini API
- LangChain
- ChromaDB
- Streamlit

## Contact
Your Name - shilpgohil@gmail.com
