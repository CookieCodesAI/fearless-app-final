import React, { useState } from 'react';
import {useNavigate} from "react-router-dom";
import "./component-styles/ProfilePic.css"

export default function ProfilePic(){
    const [open, setOpen] = useState(false);
    return(
        <div className = "settings">
            <button className="profilePic" onClick = {() => {setOpen(!open)}}/>
                {open &&(
                    <div className = "options">
                        <div>Options</div>
                        <div>Activity</div>
                        <div>Account Info</div>
                    </div>
                )}
        </div>
    )
}