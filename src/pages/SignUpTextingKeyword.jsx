import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './pages-styles/SignUpForm.css';
import './pages-styles/SignUpTextingKeyword.css';

export default function SignUpTextingKeyword() {
  const navigate = useNavigate();
  const location = useLocation();
  const prevState = location.state || {};
  const [keyword, setKeyword] = useState('');

  function handleNext() {
    if (keyword.trim()) {
      navigate('/form/callkeywords', { state: { ...prevState, keyword } });
    }
  }

  return (
    <div className="signup-page">
      <img src="/flower.png" alt="flower" className="fearless-flower" />

      <h1 className="signup-headline">Preferences</h1>

      <div className="signup-form">
        <input
          className="signup-input"
          type="text"
          placeholder="TEXTING KEYWORD"
          value={keyword}
          onChange={(e) => setKeyword(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleNext()}
          autoFocus
        />

        <p className="keyword-hint">
          Please choose a simple, non-conspicuous word you can use to send an
          SOS in our fake message system. You can always change this later.
        </p>

        <button className="signup-next-btn" onClick={handleNext}>
          NEXT. <span className="signup-arrow">→</span>
        </button>
      </div>

      <div className="fearless-blobs">
        <div className="blob-lightpink" />
        <div className="blob-lavender" />
        <div className="blob-hotpink" />
      </div>

      <div className="fearless-badge">@wearefearless</div>
    </div>
  );
}
