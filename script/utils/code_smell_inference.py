import json
import csv
import logging
import os
from typing import Optional
from tqdm import tqdm

# Import for ollama
import ollama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler('inference_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ollama model configuration
OLLAMA_MODEL = "llama3"  # You can change this to mistral, codellama, etc.

MAX_CHARS = 20000
SMELL_TYPES = ["feature_envy", "long_method", "blob", "data_class"]

# Few-shot examples for prompts
EXAMPLES = [
    {
        "code_snippet": "  @Override\n  public boolean incrementToken() throws IOException {\n    for(;;) {\n\n      if (!remainingTokens.isEmpty()) {\n        // clearAttributes();  // not currently necessary\n        restoreState(remainingTokens.removeFirst());\n        return true;\n      }\n\n      if (!input.incrementToken()) return false;\n\n      int len = termAtt.length();\n      if (len==0) return true; // pass through zero length terms\n      \n      int firstAlternativeIncrement = inject ? 0 : posAtt.getPositionIncrement();\n\n      String v = termAtt.toString();\n      String primaryPhoneticValue = encoder.doubleMetaphone(v);\n      String alternatePhoneticValue = encoder.doubleMetaphone(v, true);\n\n      // a flag to lazily save state if needed... this avoids a save/restore when only\n      // one token will be generated.\n      boolean saveState=inject;\n\n      if (primaryPhoneticValue!=null && primaryPhoneticValue.length() > 0 && !primaryPhoneticValue.equals(v)) {\n        if (saveState) {\n          remainingTokens.addLast(captureState());\n        }\n        posAtt.setPositionIncrement( firstAlternativeIncrement );\n        firstAlternativeIncrement = 0;\n        termAtt.setEmpty().append(primaryPhoneticValue);\n        saveState = true;\n      }\n\n      if (alternatePhoneticValue!=null && alternatePhoneticValue.length() > 0\n              && !alternatePhoneticValue.equals(primaryPhoneticValue)\n              && !primaryPhoneticValue.equals(v)) {\n        if (saveState) {\n          remainingTokens.addLast(captureState());\n          saveState = false;\n        }\n        posAtt.setPositionIncrement( firstAlternativeIncrement );\n        termAtt.setEmpty().append(alternatePhoneticValue);\n        saveState = true;\n      }\n\n      // Just one token to return, so no need to capture/restore\n      // any state, simply return it.\n      if (remainingTokens.isEmpty()) {\n        return true;\n      }\n\n      if (saveState) {\n        remainingTokens.addLast(captureState());\n      }\n    }\n  }",
        "smell": "long_method",
        "severity": "moderate"
    },
    {
        "code_snippet": "@Entity\npublic class Car2 {\n  @Id\n  private String numberPlate;\n  \n  private String colour;\n  \n  private int engineSize;\n  \n  private int numberOfSeats;\n\n  public String getNumberPlate() {\n    return numberPlate;\n  }\n\n  public void setNumberPlate(String numberPlate) {\n    this.numberPlate = numberPlate;\n  }\n\n  public String getColour() {\n    return colour;\n  }\n\n  public void setColour(String colour) {\n    this.colour = colour;\n  }\n\n  public int getEngineSize() {\n    return engineSize;\n  }\n\n  public void setEngineSize(int engineSize) {\n    this.engineSize = engineSize;\n  }\n\n  public int getNumberOfSeats() {\n    return numberOfSeats;\n  }\n\n  public void setNumberOfSeats(int numberOfSeats) {\n    this.numberOfSeats = numberOfSeats;\n  }\n  \n  \n}",
        "smell": "data_class",
        "severity": "moderate"
    },
    {
        "code_snippet": "@Override\n      public void read(org.apache.thrift.protocol.TProtocol prot, cancelCompaction_args struct) throws org.apache.thrift.TException {\n        org.apache.thrift.protocol.TTupleProtocol iprot = (org.apache.thrift.protocol.TTupleProtocol) prot;\n        java.util.BitSet incoming = iprot.readBitSet(2);\n        if (incoming.get(0)) {\n          struct.login = iprot.readBinary();\n          struct.setLoginIsSet(true);\n        }\n        if (incoming.get(1)) {\n          struct.tableName = iprot.readString();\n          struct.setTableNameIsSet(true);\n        }\n      }",
        "smell": "feature_envy",
        "severity": "minor"
    }
]

def create_prompt(examples):
    """
    Create the analysis prompt with few-shot examples.
    """
    prompt = "You are a code analysis expert. Analyze the following code snippet and identify if it contains any of these code smells:\n"
    prompt += f'{", ".join(SMELL_TYPES)}\n\n'
    prompt += "Examples:\n"

    for i, example in enumerate(examples, start=1):
        prompt += f"\nExample {i}:\n"
        prompt += f"Code snippet: {example['code_snippet'][:200]}...\n"
        prompt += f"Smell: {example['smell']}\n"

    prompt += "\nRespond with ONLY the smell name (one of the categories above).\n"
    return prompt


def truncate_snippet(snippet: str) -> str:
    """Truncate the code snippet to stay within the character limit."""
    return snippet[:MAX_CHARS] if len(snippet) > MAX_CHARS else snippet


def predict_smell(code_snippet: str, retries: int = 3) -> Optional[str]:
    """
    Use GPT-4 to predict the code smell.
    Ollama to predict the code smell.
    
    Args:
        code_snippet: The code snippet to analyze
        retries: Number of retry attempts
    
    Returns:
        Predicted smell name or None if failed
    """
    code_snippet = truncate_snippet(code_snippet)
    
    system_prompt = create_prompt(EXAMPLES)
    user_prompt = f"Code snippet:\n\n```\n{code_snippet}\n```\n\nWhat is the code smell?"
    
    for attempt in range(retries):
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": 0.3,
                    "num_predict": 20
                }
            )

            predicted_smell = response['message']['content']
            # Clean up the response to extract only smell type
            for smell in SMELL_TYPES:
                if smell in predicted_smell:
                    return smell
            
            # If no recognized smell, return the response as-is
            logger.warning(f"Unrecognized smell prediction: {predicted_smell}")
            return predicted_smell

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == retries - 1:
                logger.error(f"Failed to predict smell after {retries} attempts")
                return None
    
    return None


def load_json_data(json_file: str) -> list:
    """Load code snippets from JSON file."""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"JSON file not found: {json_file}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format: {json_file}")
        return []


def save_results_to_csv(results: list, csv_file: str) -> None:
    """Save inference results to CSV file."""
    try:
        with open(csv_file, 'w', newline='') as f:
            fieldnames = ['id', 'smell', 'predicted_smell']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(results)
        
        logger.info(f"Results saved to {csv_file}")
    except Exception as e:
        logger.error(f"Failed to save results to CSV: {str(e)}")


def main():
    """Main inference pipeline."""
    json_file = "MLCQCodeSmellSamples.json"
    csv_file = "CodeSmellPredictions.csv"
    
    logger.info("Loading code snippets from JSON...")
    snippets = load_json_data(json_file)
    
    if not snippets:
        logger.error("No snippets loaded. Exiting.")
        return
    
    logger.info(f"Loaded {len(snippets)} code snippets")
    
    results = []
    
    # Process each snippet
    for snippet in tqdm(snippets, desc="Processing code snippets"):
        snippet_id = snippet.get('id', 'unknown')
        actual_smell = snippet.get('smell', 'unknown')
        code = snippet.get('code_snippet', '')
        
        # Predict code smell
        predicted_smell = predict_smell(code)
        results.append({
            'id': snippet_id,
            'smell': actual_smell,
            'predicted_smell': predicted_smell if predicted_smell else 'unknown'
        })
        
        logger.info(f"ID {snippet_id}: Actual={actual_smell}, Predicted={predicted_smell}")
    
    # Save results to CSV
    logger.info("Saving results to CSV...")
    save_results_to_csv(results, csv_file)
    
    logger.info(f"Inference complete. Results saved to {csv_file}")


if __name__ == '__main__':
    main()
