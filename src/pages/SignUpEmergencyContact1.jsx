import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './pages-styles/SignUpForm.css';

export default function SignUpEmergencyContact1() {
  const navigate = useNavigate();
  const location = useLocation();
  const prevState = location.state || {};
  const [ec1Name, setEc1Name] = useState('');
  const [ec1Phone, setEc1Phone] = useState('');

  function handleNext() {
    if (ec1Name.trim() && ec1Phone.trim()) {
      navigate('/form/ec2', { state: { ...prevState, ec1Name, ec1Phone } });
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
          placeholder="EMERGENCY CONTACT 1 FULL NAME"
          value={ec1Name}
          onChange={(e) => setEc1Name(e.target.value)}
          autoFocus
        />

        <input
          className="signup-input"
          type="tel"
          placeholder="EMERGENCY CONTACT 1 MOBILE PHONE"
          value={ec1Phone}
          onChange={(e) => setEc1Phone(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleNext()}
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
