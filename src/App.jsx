<<<<<<< Updated upstream
import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <div>
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>Vite + React</h1>
      <div className="card">
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
        <p>
          Edit <code>src/App.jsx</code> and save to test HMR
        </p>
      </div>
      <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p>
    </>
  )
}

export default App
=======
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

import SignUp from "./pages/SignUp.jsx"

function App() { 
  delete L.Icon.Default.prototype._getIconUrl; 
  L.Icon.Default.mergeOptions({ 
    iconRetinaUrl: icon2x, 
    iconUrl: icon, 
    shadowUrl: shadow,
  })
  return ( 
    <Routes>
      <Route path="/*" element={
        <SignUp/>
      }/>
      <Route path="/home" element={
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
>>>>>>> Stashed changes
