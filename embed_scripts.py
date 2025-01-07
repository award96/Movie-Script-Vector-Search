import polars as pl
from transformers import AutoTokenizer, AutoModel
import torch

def script_to_embedding(
        script: str,
        tokenizer: AutoTokenizer,
        model: AutoModel,
        device: torch.device
    ) -> torch.Tensor:
    """
    String to mean pooled embedding using model.

    Args:
        script (str): The entire script as a string.
        tokenizer (AutoTokenizer): A tokenizer from Transformers.
        model (AutoModel): A model from Transformers.
        device (torch.device): CPU or GPU.

    Returns:
        torch.Tensor: A mean pooled (hidden_size,) vector representation of the script.
    """
    # tokenized into chunks
    # tokenized_script.input_ids has shape 
    # (n_chunks, model_max_length, model_hidden_size)
    tokenized_script = tokenizer(
        script,
        max_length=tokenizer.model_max_length,
        truncation=True,
        padding=True,
        add_special_tokens=True,
        return_overflowing_tokens=True,
        return_tensors='pt'
    )

    # put these 3 in input dictionary
    # and to device
    inputs = {
        key: tokenized_script[key].to(device) for key in [
            "input_ids",
            "attention_mask",
            "token_type_ids"
        ]
    }

    with torch.no_grad():
        # shape (n_batches, model_max_length, model_hidden_size)
        outputs = model(**inputs)
        
    # average over the batches, then over the tokens
    mean_pool_hidden_state = outputs.last_hidden_state.mean(dim=(0, 1))
    return mean_pool_hidden_state


if __name__ == "__main__":
    print("Loading model & tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("Starting")
    df = pl.read_parquet(
        "data/out/movie-script-dataset.parquet",
        columns=["script"]
    )

    embeddings = torch.zeros((len(df), model.config.hidden_size), dtype=torch.float32)
    print("Embedding Tensore shape: ", embeddings.shape)

    for idx in range(len(df)):
        print("On script#: ", idx)
        embedded_script = script_to_embedding(
            df[idx, "script"],
            tokenizer,
            model,
            device
        )
        embeddings[idx] = embedded_script
    
    print("Saving...")
    torch.save(embeddings, "data/out/scripts-embedded.pt")
    print("Done")