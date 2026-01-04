import { Routes, Route } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import Layout from './components/Layout';
import ProjectsPage from './pages/ProjectsPage';
import ProjectPage from './pages/ProjectPage';
import MatchingPage from './pages/MatchingPage';
import ResultsPage from './pages/ResultsPage';

function App() {
  return (
    <Layout>
      <AnimatePresence mode="wait">
        <Routes>
          <Route path="/" element={<ProjectsPage />} />
          <Route path="/project/:id" element={<ProjectPage />} />
          <Route path="/project/:id/matching" element={<MatchingPage />} />
          <Route path="/project/:id/results" element={<ResultsPage />} />
        </Routes>
      </AnimatePresence>
    </Layout>
  );
}

export default App;


