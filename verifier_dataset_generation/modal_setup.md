1. Claim the Modal credits from the email they sent us
2. Create a venv or whatever and run `pip install modal` then `modal setup`
3. Get your HF account token (assuming your account was already granted access to the Llama repos) and run `modal secret create hf-token HF_TOKEN=your_token_here`
4. To run the script, you can use `modal run llama_on_modal.py`. You can look at the code to see how it works
   - I need to finish editing it to make it more efficient, but note if you want to do scripts for this you should create volumes to save model weights and datasets into to prevent having to redownload those every run
