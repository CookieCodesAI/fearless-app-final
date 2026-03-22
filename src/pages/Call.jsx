import React, {useState, useEffect, useRef} from 'react';
import { useNavigate } from 'react-router-dom';
import "./pages-styles/Call.css";
import {useReactMediaRecorder } from 'react-media-recorder';

export default function Call(){
    const [prediction, setPrediction] = useState("");
    const [confidence, setConfidence] = useState(0);
    const [recording, setRecording] = useState(false);
    const [status, setStatus] = useState("");

    const [recorder, setRecorder] = useState(null);
    const [stream, setStream] = useState(null);

    const navigate = useNavigate();

    const sending = useRef(false);
    const queue = useRef([]);

    const sendChunk = async (blob) => {
       
        queue.current.push(blob);
        if (sending.current) return;
        sending.current = true;

        while (queue.current.length > 0) {
            const current = queue.current.shift();
            const formData = new FormData();
            formData.append("chunk", current);

            try {
                const res = await fetch("http://192.168.86.47:8000/predict", {
                    method: "POST",
                    body: formData,
                    mode: "cors",
                });
                if (!res.ok) throw new Error(`Status ${res.status}`);
                const data = await res.json();
                setPrediction(data.prediction);
                setConfidence(data.confidence);
                if (data.status.includes("NO SOS DETECTED")) console.log("NO SOS");
                else{
                    console.log(data.status);
                }
            } catch (err) {
                console.error("Chunk send failure:", err);
            }
        }

        sending.current = false;
    };


    const handleStart = async () => {
        try {
        const s = await navigator.mediaDevices.getUserMedia({ audio: true });
        setStream(s);

        const r = new MediaRecorder(s, { mimeType: "audio/webm;codecs=opus" });

        r.ondataavailable = (e) => {
            console.log("Chunk ready:", e.data);
            sendChunk(e.data);
        };

        r.start(4000); 
        setRecorder(r);
        setRecording(true);
        console.log("Recording started");
        } catch (err) {
        console.error("Microphone error:", err);
        }
    };
    const fetchTestFile = async () => {
        try {
            const res = await fetch("http://127.0.0.1:8000/test_file");
            const data = await res.json();

            for (let i = 0; i < data.length; i++) {
                const chunk = data[i];
                setPrediction(chunk.prediction);
                setConfidence(chunk.confidence);
                console.log(chunk.prediction, chunk.confidence, chunk.status);
                if (chunk.status == "SOS DETECTED SENDING HELP"){
                    setStatus(chunk.status); 
                    break;
                }
                else{
                    setStatus("");
                }
                await new Promise(res => setTimeout(res, 500)); 
            }
        } catch (err) {
            console.error(err);
        }
    };

    const handleStop = () => {
        if (recorder) recorder.stop();
        if (stream) stream.getTracks().forEach((t) => t.stop());
        setRecording(false);
        console.log("Recording stopped");
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
            }}/>
        );
    });
    return (
        <div>
            <div className="callContainer">
                {stars}
                <div className ="caller">572-299-2934</div>
                <div className = "caller-name">Krish</div>
                <button className="end-call" onClick={()=>{
                    navigate('/');
                    handleStop();
                }}>.</button>
                <button className="mute-call">.</button>
                <button className="speaker" onClick={fetchTestFile}>.</button>
                <div className = "prediction">Prediction: {prediction}</div>
                <div className = "confidence">Confidence: {Number(confidence).toFixed(2)}</div>
                <div className="status">{status}</div>
            </div>
        </div>
    )
}