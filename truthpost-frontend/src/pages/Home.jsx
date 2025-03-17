import { useState } from 'react';
import axios from 'axios';

function Home() {
  const [formData, setFormData] = useState({ title: '', content: '', media: null });

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
      await axios.post('/api/articles', formDataToSend);
      alert('Article submitted successfully!');
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="p-6 max-w-md mx-auto">
      <h1 className="text-2xl font-bold mb-4">Submit News</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        <input type="text" name="title" placeholder="Title" onChange={handleChange} className="w-full p-2 border rounded" />
        <textarea name="content" placeholder="Content" onChange={handleChange} className="w-full p-2 border rounded"></textarea>
        <input type="file" name="media" onChange={handleFileChange} className="w-full p-2 border rounded" />
        <button type="submit" className="w-full p-2 bg-blue-500 text-white rounded">Submit</button>
      </form>
    </div>
  );
}

export default Home;
