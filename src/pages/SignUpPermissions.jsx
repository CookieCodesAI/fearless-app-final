import { useNavigate, useLocation } from 'react-router-dom';
import './pages-styles/SignUpForm.css';
import './pages-styles/SignUpPermissions.css';

export default function SignUpPermissions() {
  const navigate = useNavigate();
  const location = useLocation();
  const prevState = location.state || {};

  async function requestPermissions() {
    try {
      // Request camera and microphone
      await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    } catch (e) {
      console.log('Camera/mic permission denied or unavailable:', e);
    }

    try {
      // Request location
      navigator.geolocation.getCurrentPosition(
        () => {},
        (e) => console.log('Location permission denied:', e)
      );
    } catch (e) {
      console.log('Geolocation unavailable:', e);
    }
  }

  function handleNext() {
    navigate('/form/keyword', { state: { ...prevState } });
  }

  return (
    <div className="signup-page" ref={(el) => { if (el) requestPermissions(); }}>
      <img src="/flower.png" alt="flower" className="fearless-flower" />

      <h1 className="signup-headline">Sign Up</h1>

      <div className="signup-form">
        <p className="permissions-text">
          Please allow us access to your{' '}
          <strong>microphone</strong>,{' '}
          <strong>camera</strong>,{' '}
          and{' '}
          <strong>location</strong>.{' '}
          If you don't see a pop-up,{' '}
          <span
            className="permissions-link"
            onClick={requestPermissions}
          >
            click here
          </span>
          .
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
