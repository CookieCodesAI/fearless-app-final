import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './pages-styles/SignUpForm.css';

export default function SignUpForm() {
  const navigate = useNavigate();
  const [firstName, setFirstName] = useState('');

  function handleNext() {
    if (firstName.trim()) {
      navigate('lastname', { state: { firstName } });
    }
  }

  return (
    <div className="signup-page">
      <img src="/flower.png" alt="flower" className="fearless-flower" />

      <h1 className="signup-headline">Sign Up</h1>

      <div className="signup-form">
        <input
          className="signup-input"
          type="text"
          placeholder="FIRST NAME"
          value={firstName}
          onChange={(e) => setFirstName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleNext()}
          autoFocus
        />

        <button className="signup-next-btn" onClick={handleNext}>
          NEXT. <span className="signup-arrow">→</span>
        </button>

        <p className="signup-legal">
          BY CONTINUING, YOU AGREE TO OUR{' '}
          <span className="signup-legal-bold">PRIVACY POLICY</span>.
        </p>
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
