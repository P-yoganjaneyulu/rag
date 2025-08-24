# ✅ Deployment Checklist for Render

## Before Deploying

### 1. **Code Repository** ✅
- [ ] All code is committed and pushed to Git repository
- [ ] Repository is public or Render has access
- [ ] Main branch contains the latest code

### 2. **Required Files** ✅
- [ ] `app/app_render.py` - Production-ready Streamlit app
- [ ] `chatbot/rag_pipeline_simple.py` - Vector store builder
- [ ] `data/sdg.csv` - SDG dataset
- [ ] `requirements.txt` - Python dependencies
- [ ] `render.yaml` - Render configuration
- [ ] `Dockerfile` - Alternative deployment option

### 3. **Dependencies** ✅
- [ ] All packages listed in `requirements.txt`
- [ ] No missing imports or dependencies
- [ ] Compatible Python version (3.9+)

### 4. **Data Files** ✅
- [ ] `data/sdg.csv` is included in repository
- [ ] File size is reasonable (< 100MB)
- [ ] No sensitive data in the dataset

## Deployment Steps

### 1. **Render Setup**
- [ ] Create Render account at [render.com](https://render.com)
- [ ] Connect your Git repository
- [ ] Choose deployment method (Web Service, Blueprint, or Docker)

### 2. **Service Configuration**
- [ ] **Name**: `sdg-insight-chatbot`
- [ ] **Environment**: `Python 3`
- [ ] **Build Command**: `pip install -r requirements.txt`
- [ ] **Start Command**: `streamlit run app/app_render.py --server.port $PORT --server.address 0.0.0.0`
- [ ] **Plan**: Free (or choose paid plan)

### 3. **Environment Variables** (Optional)
- [ ] `PYTHON_VERSION`: `3.9.16`

### 4. **Deploy**
- [ ] Click "Create Web Service"
- [ ] Monitor build process
- [ ] Check for any build errors
- [ ] Wait for deployment to complete

## Post-Deployment

### 1. **Testing**
- [ ] App loads without errors
- [ ] Vector store builds automatically on first run
- [ ] Chatbot responds to questions
- [ ] No timeout or memory issues

### 2. **Monitoring**
- [ ] Check Render dashboard for logs
- [ ] Monitor resource usage
- [ ] Set up alerts if needed

### 3. **Optimization**
- [ ] Consider upgrading from free tier if needed
- [ ] Set up custom domain (paid plans)
- [ ] Configure auto-deploy from main branch

## Troubleshooting

### Common Issues:
- **Build failures**: Check `requirements.txt` and dependencies
- **Memory issues**: Free tier limitations; consider upgrading
- **Timeout issues**: Vector store building takes time on first run
- **Port issues**: Ensure app uses `$PORT` environment variable

### Debug Steps:
1. Check Render build logs
2. Verify all files are in repository
3. Test locally first
4. Check Streamlit compatibility

## Success Indicators ✅

- [ ] App deploys successfully
- [ ] Vector store builds automatically
- [ ] Chatbot responds to queries
- [ ] No critical errors in logs
- [ ] App is accessible via Render URL

## Next Steps

After successful deployment:
1. **Share the URL** with users
2. **Monitor performance** and usage
3. **Consider upgrades** for production use
4. **Set up CI/CD** for automatic deployments
5. **Add analytics** and monitoring

---

**🎉 Congratulations!** Your SDG Insight Chatbot is now live on Render!
