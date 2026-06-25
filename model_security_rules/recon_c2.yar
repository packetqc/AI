/*
 * recon_c2.yar — editable YARA rules for blackbox model forensics.
 *
 * Security engineers: ADD / TUNE rules here. The analyzer
 * (model_security_re.py) compiles this file and matches it against the staged
 * GGUF blob, writing hits to the forensic case folder (binary_forensics.md).
 *
 * Point the tool at custom rules with:  --yar /path/to/your.yar
 *
 * Conventions: set meta.severity = info|low|medium|high|critical and
 * meta.description so the report renders your intent. Strings are matched
 * against the raw model file (tokens are stored as text in GGUF metadata).
 */

rule recon_grammar_vocabulary
{
    meta:
        description = "Network-reconnaissance action vocabulary embedded in the model"
        severity    = "high"
        author      = "model-security-re (editable)"
    strings:
        $nmap   = "nmap" nocase
        $scan   = "scan" nocase
        $sweep  = "sweep" nocase
        $detect = "detect" nocase
        $ports  = "port" nocase
        $disc   = "discovery" nocase
        $arp    = "arp" nocase
        $uname  = "uname" nocase
        $svc    = "svc" nocase
    condition:
        4 of them
}

rule c2_network_indicator
{
    meta:
        description = "Possible C2 / network endpoint indicators (URL, IP, onion)"
        severity    = "critical"
        author      = "model-security-re (editable)"
    strings:
        $url   = /https?:\/\/[a-zA-Z0-9.\-]{4,}/ nocase
        $onion = ".onion" nocase
        $ipv4  = /([0-9]{1,3}\.){3}[0-9]{1,3}/
        $beacon = "beacon" nocase
        $exfil = "exfil" nocase
    condition:
        any of them
}

rule shell_exec_signature
{
    meta:
        description = "Direct shell / code-execution signatures"
        severity    = "critical"
        author      = "model-security-re (editable)"
    strings:
        $sh   = "/bin/sh"
        $bash = "/bin/bash"
        $sub  = "subprocess"
        $osys = "os.system"
        $pwsh = "powershell" nocase
        $eval = "eval("
        $exec = "exec("
    condition:
        any of them
}

rule encoded_blob_suspect
{
    meta:
        description = "Long base64/hex run — candidate hidden/encoded payload"
        severity    = "medium"
        author      = "model-security-re (editable)"
    strings:
        $b64 = /[A-Za-z0-9+\/]{40,}={0,2}/
        $hex = /[0-9a-fA-F]{48,}/
    condition:
        any of them
}
