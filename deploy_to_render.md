# 🚀 Deploy SDG Insight Chatbot to Render

## Prerequisites
- A Render account (free tier available)
- Your project pushed to a Git repository (GitHub, GitLab, etc.)

## Deployment Steps

### Option 1: Using Render Dashboard (Recommended)

1. **Sign up/Login to Render**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub or create an account

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your Git repository
   - Select the repository containing this project

3. **Configure the Service**
   - **Name**: `sdg-insight-chatbot` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app/app_simple.py --server.port $PORT --server.address 0.0.0.0`
   - **Plan**: Free (or choose paid plan for better performance)

4. **Environment Variables** (Optional)
   - `PYTHON_VERSION`: `3.9.16`

5. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your app

### Option 2: Using render.yaml (Infrastructure as Code)

1. **Push your code** with the `render.yaml` file to your repository
2. **In Render Dashboard**:
   - Go to "Blueprints"
   - Click "New Blueprint Instance"
   - Connect your repository
   - Render will automatically create the service based on `render.yaml`

### Option 3: Using Docker

1. **Push your code** with the `Dockerfile` to your repository
2. **In Render Dashboard**:
   - Create new "Web Service"
   - Select "Docker" as environment
   - Render will use your `Dockerfile` automatically

## Important Notes

### ⚠️ Vector Store Issue
The current setup creates the vector store locally. For production deployment, you'll need to:

1. **Pre-build the vector store** and include it in your repository, OR
2. **Modify the app** to build the vector store during deployment

### 🔧 Recommended Fix for Production
Modify `app/app_simple.py` to automatically build the vector store if it doesn't exist:

```python
# Add this function to automatically build vector store if needed
def ensure_vectorstore():
    if not os.path.exists("vectorstore/faiss_index.index"):
        st.info("Building vector store... This may take a few minutes on first run.")
        import subprocess
        subprocess.run(["python", "chatbot/rag_pipeline_simple.py"])
        st.success("Vector store built successfully!")
```

### 📁 File Structure for Deployment
Ensure your repository includes:
```
├── app/
│   └── app_simple.py
├── chatbot/
│   └── rag_pipeline_simple.py
├── data/
│   └── sdg.csv
├── requirements.txt
├── render.yaml (or Dockerfile)
└── README.md
```

## Post-Deployment

1. **Test your app** at the provided Render URL
2. **Monitor logs** in the Render dashboard
3. **Set up custom domain** if needed (paid plans)
4. **Configure auto-deploy** from your main branch

## Troubleshooting

### Common Issues:
- **Build failures**: Check if all dependencies are in `requirements.txt`
- **Port issues**: Ensure the app uses `$PORT` environment variable
- **Memory issues**: Free tier has limitations; consider upgrading
- **Vector store**: Ensure the data files are included in your repository

### Support:
- Check Render's [documentation](https://render.com/docs)
- Review build logs in Render dashboard
- Check Streamlit's [deployment guide](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)

## Cost Estimation

- **Free Tier**: $0/month (limited resources, auto-sleep after inactivity)
- **Starter Plan**: $7/month (always on, better performance)
- **Standard Plan**: $25/month (dedicated resources, custom domains)

For a production chatbot, consider the Starter or Standard plan for better reliability and performance.
