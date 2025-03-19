import { useState } from 'react';
import axios from 'axios';

function Home() {
  const [formData, setFormData] = useState({ title: '', content: '', media: null });
  const [result, setResult] = useState(null); // to show response result

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleFileChange = (e) => {
    setFormData({ ...formData, media: e.target.files[0] });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formDataToSend = new FormData();
    formDataToSend.append('title', formData.title);
    formDataToSend.append('content', formData.content);
    if (formData.media) formDataToSend.append('media', formData.media);

    try {
      const res = await axios.post('/api/articles', formDataToSend);

      // Store result for display
      setResult({
        text: res.data.textAnalysis,
        media: res.data.mediaAnalysis,
        finalStatus: res.data.finalStatus,
      });
    } catch (error) {
      console.error(error);
      alert('Something went wrong while submitting.');
    }
  };

  return (
    <div className="p-6 max-w-md mx-auto">
      <h1 className="text-2xl font-bold mb-4">Submit News</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        <input
          type="text"
          name="title"
          placeholder="Title"
          onChange={handleChange}
          className="w-full p-2 border rounded"
        />
        <textarea
          name="content"
          placeholder="Content"
          onChange={handleChange}
          className="w-full p-2 border rounded"
        ></textarea>
        <input
          type="file"
          name="media"
          accept="image/*,video/mp4"
          onChange={handleFileChange}
          className="w-full p-2 border rounded"
        />
        <small className="text-gray-500">Supported formats: JPG, PNG, MP4</small>
        <button type="submit" className="w-full p-2 bg-blue-500 text-white rounded">
          Submit
        </button>
      </form>

      {result && (
        <div className="mt-6 p-4 border rounded bg-gray-50 text-sm">
          <h2 className="text-lg font-semibold mb-2">Scan Results:</h2>

          {/* Text Result */}
          {result.text ? (
            <p>
              <strong>📝 Text:</strong> {result.text.label} ({(result.text.confidence * 100).toFixed(2)}% confidence)
            </p>
          ) : (
            <p><strong>📝 Text:</strong> Not scanned</p>
          )}

          {/* Media Result */}
          {result.media ? (
            <p>
              <strong>🖼️ Media:</strong> {result.media.label} ({(result.media.confidence * 100).toFixed(2)}% confidence)
            </p>
          ) : (
            <p><strong>🖼️ Media:</strong> Not scanned</p>
          )}

          {/* Final Decision */}
          <p className={`mt-2 font-semibold ${result.finalStatus === 'approved' ? 'text-green-600' : 'text-red-600'}`}>
            Final Decision: {result.finalStatus.toUpperCase()}
          </p>
        </div>
      )}
    </div>
  );
}

export default Home;
