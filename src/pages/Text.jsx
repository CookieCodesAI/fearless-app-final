import React, { useState } from "react";
import { useNavigate } from 'react-router-dom';
import "./pages-styles/Text.css";

export default function Text() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const navigate = useNavigate();
  const responses = {
    hello: "Hi there!",
    "how are you": "I'm just a bot, but I'm good!",
    bye: "Goodbye! See you later.",
    blueberry: "Sending SOS...",
  };

  const sendMessage = () => {
    if (!input) return;

    setMessages((prev) => [...prev, { sender: "You", text: input }]);
    let foundResponse = "Sorry, I don't understand that.";
    for (const key in responses) {
      if (input.toLowerCase().includes(key)) {
        foundResponse = responses[key];
        break;
      }
    }
    

    setMessages((prev) => [...prev, { sender: "Krish", text: foundResponse }]);
    setInput("");
  };
  const stars = Array.from({length:175}).map((_,index)=>{
        const top = Math.random()*100;
        const left = Math.random()*100;
        const delay = Math.random()*5+1;
        const size = Math.random()*3;
        return(
            <span key = {index} className = 'star' style = {{
                top : `${top}%`,
                left : `${left}%`,
                animationDelay : `${delay}s`,
                width : `${size}px`,
                height : `${size}px`,
                zIndex : 0,
            }}/>
        );
    });

  return (
    <div className="chat-container">
    <button className = "return" onClick={()=>navigate('/home')}>↰</button>
    <div className="profile-pic"></div>
     <div className="contact">Krish</div>
      <div className="chat-log">
        {messages.map((msg, index) => (
          <div key={index} className="chat-message">
            <strong>{msg.sender}: </strong>
            {msg.text}
          </div>
        ))}
      </div>
      <input
        className="chat-input"
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && sendMessage()}
      />
      <button className="chat-button" onClick={sendMessage}>
        ↑ 
      </button>
      {stars}
    </div>
  );
}