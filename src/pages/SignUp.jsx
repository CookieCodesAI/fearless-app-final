import { Routes, Route, useNavigate } from 'react-router-dom';
import WelcomePage from './WelcomePage';
import SignUpForm from './SignUpForm';
import SignUpLastName from './SignUpLastName';
import SignUpPhone from './SignUpPhone';
import SignUpEmergencyContact1 from './SignUpEmergencyContact1';
import SignUpEmergencyContact2 from './SignUpEmergencyContact2';
import SignUpPassword from './SignUpPassword';
import SignUpPermissions from './SignUpPermissions';
import SignUpTextingKeyword from './SignUpTextingKeyword';
import SignUpCallKeywords from './SignUpCallKeywords';

export default function SignUp() {
  return (
    <Routes>
      <Route path="/" element={<WelcomePage />} />
      <Route path="form" element={<SignUpForm />} />
      <Route path="form/lastname" element={<SignUpLastName />} />
      <Route path="form/phone" element={<SignUpPhone />} />
      <Route path="form/ec1" element={<SignUpEmergencyContact1 />} />
      <Route path="form/ec2" element={<SignUpEmergencyContact2 />} />
      <Route path="form/password" element={<SignUpPassword />} />
      <Route path="form/permissions" element={<SignUpPermissions />} />
      <Route path="form/keyword" element={<SignUpTextingKeyword />} />
      <Route path="form/callkeywords" element={<SignUpCallKeywords />} />
      
    </Routes>
  );
}
