import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './pages-styles/SignUpForm.css';

export default function SignUpPhone() {
  const navigate = useNavigate();
  const location = useLocation();
  const prevState = location.state || {};
  const [phone, setPhone] = useState('');

  function handleNext() {
    if (phone.trim()) {
      navigate('/form/ec1', { state: { ...prevState, phone } });
    }
  }

  return (
    <div className="signup-page">
      <img src="/flower.png" alt="flower" className="fearless-flower" />

      <h1 className="signup-headline">Sign Up</h1>

      <div className="signup-form">
        <input
          className="signup-input"
          type="tel"
          placeholder="PHONE NUMBER"
          value={phone}
          onChange={(e) => setPhone(e.target.value)}
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
