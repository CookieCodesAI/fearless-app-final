import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './pages-styles/SignUpForm.css';

export default function SignUpLastName() {
  const navigate = useNavigate();
  const location = useLocation();
  const firstName = location.state?.firstName || '';
  const [lastName, setLastName] = useState('');

  function handleNext() {
    if (lastName.trim()) {
      navigate('/form/phone', { state: { firstName, lastName } });
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
          placeholder="LAST NAME"
          value={lastName}
          onChange={(e) => setLastName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleNext()}
          autoFocus
        />

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
