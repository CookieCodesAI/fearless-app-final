import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './pages-styles/SignUpForm.css';

export default function SignUpEmergencyContact2() {
  const navigate = useNavigate();
  const location = useLocation();
  const prevState = location.state || {};
  const [ec2Name, setEc2Name] = useState('');
  const [ec2Phone, setEc2Phone] = useState('');

  function handleNext() {
    if (ec2Name.trim() && ec2Phone.trim()) {
      navigate('/form/password', { state: { ...prevState, ec2Name, ec2Phone } });
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
          placeholder="EMERGENCY CONTACT 2 FULL NAME"
          value={ec2Name}
          onChange={(e) => setEc2Name(e.target.value)}
          autoFocus
        />

        <input
          className="signup-input"
          type="tel"
          placeholder="EMERGENCY CONTACT 2 MOBILE PHONE"
          value={ec2Phone}
          onChange={(e) => setEc2Phone(e.target.value)}
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
