import { useEffect, useState } from 'react';
import axios from 'axios';

function Admin() {
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

  const updateStatus = async (id, status) => {
    try {
      await axios.put(`/api/articles/${id}`, { status });
      setArticles((prevArticles) =>
        prevArticles.map((article) =>
          article._id === id ? { ...article, status } : article
        )
      );
    } catch (error) {
      console.error('Error updating status:', error);
    }
  };

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Admin Panel</h1>
      {articles.length === 0 ? (
        <p>No articles to review.</p>
      ) : (
        articles.map((article) => (
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

            <p className={`mt-2 text-sm font-medium ${article.status === 'approved' ? 'text-green-500' : 'text-red-500'}`}>
              Status: {article.status}
            </p>
            {article.status === 'pending' && (
              <div className="mt-2 space-x-2">
                <button
                  onClick={() => updateStatus(article._id, 'approved')}
                  className="px-4 py-1 bg-green-500 text-white rounded"
                >
                  Approve
                </button>
                <button
                  onClick={() => updateStatus(article._id, 'rejected')}
                  className="px-4 py-1 bg-red-500 text-white rounded"
                >
                  Reject
                </button>
              </div>
            )}
          </div>
        ))
      )}
    </div>
  );
}

export default Admin;
