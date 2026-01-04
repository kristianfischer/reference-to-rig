import { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Guitar, Waves, Home } from 'lucide-react';

interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation();

  return (
    <div className="h-screen flex flex-col tolex-texture">
      {/* Header */}
      <header className="flex-shrink-0 border-b border-amp-steel bg-amp-charcoal/80 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-3 group">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-amp-amber to-amp-orange flex items-center justify-center shadow-lg group-hover:scale-105 transition-transform">
              <Guitar className="w-6 h-6 text-amp-black" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-amp-cream tracking-tight">
                Reference-to-Rig
              </h1>
              <p className="text-xs text-amp-silver -mt-0.5">
                Tone Matching Engine
              </p>
            </div>
          </Link>

          <nav className="flex items-center gap-2">
            <NavLink to="/" active={location.pathname === '/'}>
              <Home className="w-4 h-4" />
              Projects
            </NavLink>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
          >
            {children}
          </motion.div>
        </div>
      </main>

      {/* Footer */}
      <footer className="flex-shrink-0 border-t border-amp-steel bg-amp-charcoal/50 px-6 py-3">
        <div className="max-w-7xl mx-auto flex items-center justify-between text-xs text-amp-silver">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-amp-green led-glow" />
            <span>Engine Connected</span>
          </div>
          <span>v0.1.0 MVP</span>
        </div>
      </footer>
    </div>
  );
}

function NavLink({ to, active, children }: { to: string; active: boolean; children: ReactNode }) {
  return (
    <Link
      to={to}
      className={`
        flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all
        ${active
          ? 'bg-amp-amber/20 text-amp-amber'
          : 'text-amp-silver hover:text-amp-cream hover:bg-amp-steel/50'
        }
      `}
    >
      {children}
    </Link>
  );
}


