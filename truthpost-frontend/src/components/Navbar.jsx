import { Link } from 'react-router-dom';

function Navbar() {
  return (
    <nav className="bg-gray-800 p-4 text-white">
      <div className="container mx-auto flex justify-between">
        <h1 className="text-lg font-bold">TruthPost</h1>
        <div className="space-x-4">
          <Link to="/" className="hover:underline">Home</Link>
          <Link to="/news" className="hover:underline">News</Link>
          <Link to="/admin" className="hover:underline">Admin</Link>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
