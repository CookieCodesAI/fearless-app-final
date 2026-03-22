import React from 'react';
import "./component-styles/Scrollbar.css"
import { useNavigate } from 'react-router-dom';

export default function Scrollbar(){
    const navigate = useNavigate();
    return (
        <div>
            <div className="greeting">Good Morning, Dhriti</div>
            <div className="shortcuts">Shortcuts</div>
            <div className = "scrollbar-container">
                <div className = "scrollbar">
                    <div className ="components" onClick = {() => navigate('/')}>All</div>
                    <div className ="components" onClick = {() => navigate('/location')}>Map</div>
                    <div className ="components" onClick = {() => navigate('/livestream')}>Live Stream</div>
                    <div className ="components" onClick = {() => navigate('/text')}>Text</div>
                    <div className ="components" onClick = {() => navigate('/call')}>Calls</div>
                    <div className ="components">Notes</div>
                </div>
            </div>
        </div>

    )
}