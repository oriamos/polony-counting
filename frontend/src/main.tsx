import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import './styles.css';
import { useApp } from './state/store';

// debugging / e2e hook (read-only use): window.__polony.getState()
(window as unknown as { __polony: typeof useApp }).__polony = useApp;

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
