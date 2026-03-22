import React, { useState } from 'react';
import {useNavigate} from "react-router-dom";
import "./component-styles/ProfilePic.css"

export default function ProfilePic(){
    const [open, setOpen] = useState(false);
    return(
        <div className = "settings">
            <button className="profilePic" onClick = {() => {setOpen(!open)}}></button>
                {open &&(
                    <div className = "options">
                        <button className = "op">Options</button>
                        <button className = "activity">Activity</button>
                        <button className = "accinfo">Account Info</button>
                    </div>
                )}
        </div>
    )
}