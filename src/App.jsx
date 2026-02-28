import './App.css' 
import ProfilePic from "./components/ProfilePic.jsx"
import Scrollbar from './components/Scrollbar.jsx'
import ButtonPanel from './components/ButtonPanel.jsx'
import Call from "./pages/Call.jsx"
import Text from "./pages/Text.jsx"
import {Routes, Route} from 'react-router-dom'
import LiveStream from "./pages/LiveStream.jsx"
import Location from "./pages/Location.jsx" 
import L from "leaflet"; 
import "leaflet/dist/leaflet.css"; 
import icon from "leaflet/dist/images/marker-icon.png"; 
import icon2x from "leaflet/dist/images/marker-icon-2x.png"; 
import shadow from "leaflet/dist/images/marker-shadow.png";
function App() { 
  delete L.Icon.Default.prototype._getIconUrl; 
  L.Icon.Default.mergeOptions({ 
    iconRetinaUrl: icon2x, 
    iconUrl: icon, 
    shadowUrl: shadow,
  })
  return ( 
    <Routes>
      <Route path="/" element={
        <div className='container'>
          <ProfilePic/>
          <Scrollbar/>
          <ButtonPanel/>
        </div> 
      }/>
      <Route path = '/call' element = {<Call/>} />
      {<Route path = '/text' element = {<Text/>} />}
      <Route path = '/livestream' element = {<LiveStream/>} />
      {<Route path = '/location' element = {<Location/>} />}
    </Routes>
  ) } 
export default App