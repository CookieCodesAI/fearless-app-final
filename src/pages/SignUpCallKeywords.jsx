import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './pages-styles/SignUpForm.css';
import './pages-styles/SignUpTextingKeyword.css';

export default function SignUpCallKeywords() {
  const navigate = useNavigate();
  const location = useLocation();
  const prevState = location.state || {};
  const [callKeywords, setCallKeywords] = useState('');

  function handleNext() {
    if (callKeywords.trim()) {
      navigate('/home', { state: { ...prevState, callKeywords } });
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
          placeholder="SEQUENCE OF FOUR BASIC WORDS"
          value={callKeywords}
          onChange={(e) => setCallKeywords(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleNext()}
          autoFocus
        />

        <p className="keyword-hint">
          Separate words via commas. Please choose simple words used to send an
          SOS in our fake call system. You can always change this later.
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
