import { useNavigate } from 'react-router-dom';
import './pages-styles/WelcomePage.css';

function FlowerIcon() {
  return (
    <svg
      className="fearless-flower"
      viewBox="0 0 44 44"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <circle cx="22" cy="22" r="5" fill="currentColor" opacity="0.95" />
      {[0, 45, 90, 135, 180, 225, 270, 315].map((angle, i) => (
        <ellipse
          key={i}
          cx="22"
          cy="22"
          rx="4"
          ry="9"
          fill="currentColor"
          opacity="0.75"
          transform={`rotate(${angle} 22 22) translate(0 -10)`}
          style={{ transformOrigin: '22px 22px' }}
        />
      ))}
    </svg>
  );
}

export default function WelcomePage() {
  const navigate = useNavigate();

  return (
    <div className="fearless-page">
      <img src="/flower.png" alt="flower" className="fearless-flower" />

      <p className="fearless-welcome">welcome</p>

      <h1 className="fearless-headline" onClick={() => navigate('form')}>Join Us.</h1>

      <p className="fearless-subtitle">
        and live <em>fearless</em>
      </p>

      <div className="fearless-blobs">
        <div className="blob-lightpink" />
        <div className="blob-lavender" />
        <div className="blob-hotpink" />
      </div>

      <div className="fearless-badge">@wearefearless</div>
    </div>
  );
}
