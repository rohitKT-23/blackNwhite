import { useEffect, useState } from 'react';
import axios from 'axios';

function News() {
  const [articles, setArticles] = useState([]);

  useEffect(() => {
    const fetchArticles = async () => {
      try {
        const response = await axios.get('/api/articles');
        setArticles(response.data);
      } catch (error) {
        console.error('Error fetching articles:', error);
      }
    };
    fetchArticles();
  }, []);

  // ✅ Filter only "approved" articles
  const approvedArticles = articles.filter(article => article.status === 'approved');

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Latest News</h1>
      {approvedArticles.length === 0 ? (
        <p>No approved articles available.</p>
      ) : (
        approvedArticles.map((article) => (
          <div key={article._id} className="p-4 border rounded-lg mb-4 shadow-md">
            <h2 className="text-xl font-semibold">{article.title}</h2>
            <p className="text-gray-700">{article.content}</p>
            {article.mediaUrl && (
  <>
    {article.mediaUrl.endsWith(".mp4") ? (
      <video controls className="mt-2 rounded-lg max-w-full">
        <source src={article.mediaUrl} type="video/mp4" />
        Your browser does not support the video tag.
      </video>
    ) : (
      <img
        src={article.mediaUrl}
        alt="News"
        className="mt-2 rounded-lg max-w-full"
      />
    )}
  </>
)}

            <p className="mt-2 text-sm font-medium text-green-500">
              Status: {article.status}
            </p>
          </div>
        ))
      )}
    </div>
  );
}

export default News;
