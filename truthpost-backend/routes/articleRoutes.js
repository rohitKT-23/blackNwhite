const express = require("express");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");
const Article = require("../models/Article");

const router = express.Router();

// Multer Storage Setup
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/"); // Save files in uploads/ folder
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + "-" + file.originalname); // Unique filename
  },
});

const upload = multer({ storage });

// 📝 Submit article with AI analysis
router.post("/", upload.single("media"), async (req, res) => {
  try {
    const { title, content } = req.body;
    const mediaUrl = req.file ? `/uploads/${req.file.filename}` : null;

    // Step 1: Save as "pending" in MongoDB
    const newArticle = new Article({
      title,
      content,
      mediaUrl,
      status: "pending",
      createdAt: new Date(),
    });
    await newArticle.save();

    // Step 2: Call AI services
    const textScan = await axios.post("http://localhost:8001/predict-text", { text: content });
    const textResult = textScan.data;
    let mediaResult = { label: "No Media", confidence: 0 };

    if (req.file) {
      const fileExt = req.file.originalname.split('.').pop().toLowerCase();
      const formData = new FormData();
      const fileStream = fs.createReadStream(req.file.path);
    
      if (["jpg", "jpeg", "png"].includes(fileExt)) {
        formData.append("image", fileStream);
    
        const imageScan = await axios.post("http://localhost:8002/predict-image", formData, {
          headers: { ...formData.getHeaders() },
        });
    
        mediaResult = {
          label: imageScan.data.label || "unknown",
          confidence: imageScan.data.confidence || 0,
        };
    
      } else if (["mp4", "mov", "avi"].includes(fileExt)) {
        formData.append("file", fileStream); // key must match FastAPI endpoint
    
        const videoScan = await axios.post("http://localhost:8003/predict-video", formData, {
          headers: { ...formData.getHeaders() },
        });
    
        mediaResult = {
          label: videoScan.data.deepfake_detected ? "deepfake" : "real",
          confidence: videoScan.data.confidence,
        };
      }
    }
    

    // Step 3: Determine article status
    let status = "approved";

    const textLabel = textResult.label.toLowerCase();
    const mediaLabel = mediaResult.label.toLowerCase();
    if (textLabel === "fake news" || mediaLabel === "deepfake") {
      status = "rejected";
    }

    // Step 4: Update article status
    newArticle.status = status;
    await newArticle.save();

    // Step 5: Send response
    res.json({
      message: "Article submitted successfully",
      articleId: newArticle._id,
      textAnalysis: textResult,
      mediaAnalysis: mediaResult,
      finalStatus: status,
    });

  } catch (error) {
    console.error("Error processing article:", error.message);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

// 📌 Fetch all articles
router.get("/", async (req, res) => {
  try {
    const articles = await Article.find();
    res.json(articles);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// 🔍 Trigger AI Scan (Placeholder)
router.get("/scan/:id", async (req, res) => {
  const { id } = req.params;
  res.json({ message: `Scan triggered for article ${id}` });
});

// ✅ Update article status
router.put("/:id", async (req, res) => {
  try {
    const { status } = req.body;
    const article = await Article.findByIdAndUpdate(req.params.id, { status }, { new: true });
    res.json(article);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
